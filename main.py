"""
Étape 2 : un cube fixé à la paume, qui tourne avec elle.
Mono-thread. Délégué GPU avec repli CPU (comme à l'étape 1).

Principe
  1. MediaPipe donne 21 points en pixels + 21 points "monde" (mètres, centrés sur la main).
  2. Repère de la paume :  y = poignet -> base du majeur
                           x = travers de la paume (base de l'index <-> base de l'auriculaire)
                           z = normale qui sort de la paume (z = x ^ y)
  3. Position 3D : la profondeur vient de la taille de la paume (pixels vs mètres), modèle sténopé.
  4. Cube : sommets dans le repère de la paume -> repère caméra -> projection -> faces visibles.

  5. Occlusion : quand le cube est derrière la main (dos de la main vers la caméra), on le masque
     là où il passe sous la silhouette de la main (paume remplie + doigts épaissis, tirée des points clés).

Touches : q / Échap quitter | s squelette | a axes de debug | + / - taille du cube
          o occlusion on/off | m afficher le masque de la main (debug)
Option  : --cpu pour forcer le CPU
"""
import argparse
import os
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ------------------------------------------------------------------ réglages
CUBE_SIZE = 0.05          # arête du cube (mètres)
HOVER = 0.005             # petit espace entre la paume et le cube (mètres)
FOCAL_FACTOR = 1.0        # focale (pixels) ≈ FOCAL_FACTOR x largeur de l'image
SMOOTHING = 0.5           # 0 = aucun lissage ; 0.8 = très lisse mais avec du retard
SWAP_HANDEDNESS = True   # mettre True si le cube apparaît dans le dos de la main
FINGER_THICKNESS = 0.018  # épaisseur d'un doigt (mètres), règle la silhouette utilisée pour l'occlusion

# ------------------------------------------------------------------ modèle MediaPipe
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "models", "hand_landmarker.task")
MIN_MODEL_BYTES = 1_000_000  # le vrai fichier fait ~7,5 Mo


def ensure_model():
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) < MIN_MODEL_BYTES:
        os.remove(MODEL_PATH)  # téléchargement précédent incomplet : on repart de zéro
    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        print("Téléchargement du modèle hand_landmarker (~8 Mo)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


def build_landmarker(use_gpu):
    """Tente le délégué GPU, retombe sur CPU en cas d'échec. Renvoie (landmarker, délégué utilisé)."""
    Delegate = mp_python.BaseOptions.Delegate

    def _build(delegate):
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=ensure_model(), delegate=delegate),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,  # une seule main pour cette étape
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        return vision.HandLandmarker.create_from_options(options)

    if use_gpu:
        try:
            return _build(Delegate.GPU), "GPU"
        except Exception as e:
            print(f"[avertissement] delegue GPU indisponible ({e}) -> repli sur CPU")
    return _build(Delegate.CPU), "CPU"


# ------------------------------------------------------------------ lissage
class Ema:
    """Moyenne mobile exponentielle : réduit le tremblement des points."""
    def __init__(self, alpha):
        self.alpha = alpha
        self.s = None

    def __call__(self, x):
        x = np.asarray(x, dtype=np.float64)
        self.s = x if self.s is None else self.alpha * self.s + (1.0 - self.alpha) * x
        return self.s

    def reset(self):
        self.s = None


# ------------------------------------------------------------------ pose de la paume
PALM_IDS = [0, 5, 9, 13, 17]                       # poignet + base des 4 doigts
DEPTH_SEGMENTS = [(0, 5), (0, 17), (5, 17), (0, 9)]  # segments de la paume utilisés pour la profondeur


def unit(v):
    return v / (np.linalg.norm(v) + 1e-9)


def palm_pose(px, world, label, w, h, f):
    """
    px    : (21, 2) points en pixels
    world : (21, 3) points en mètres (x droite, y bas, z devant, centrés sur la main)
    Renvoie R (3x3, colonnes = axes x, y, z de la paume dans le repère caméra) et t (centre de la paume, mètres).
    """
    # --- Orientation ---
    y = unit(world[9] - world[0])                        # vers les doigts
    right = (label == "Right") != SWAP_HANDEDNESS        # l'image est en miroir, MediaPipe en tient compte
    across = (world[17] - world[5]) if right else (world[5] - world[17])
    x = unit(across - np.dot(across, y) * y)             # on rend x perpendiculaire à y (Gram-Schmidt)
    n = np.cross(x, y)                                   # normale : sort de la paume
    R = np.column_stack([x, y, n])

    # --- Profondeur (sténopé) : taille en pixels = f * taille réelle / Z ---
    world_len = sum(np.linalg.norm(world[a, :2] - world[b, :2]) for a, b in DEPTH_SEGMENTS)
    px_len = sum(np.linalg.norm(px[a] - px[b]) for a, b in DEPTH_SEGMENTS)
    Z = float(np.clip(f * world_len / max(px_len, 1e-6), 0.1, 3.0))

    # --- Position 3D du centre de la paume ---
    u, v = px[PALM_IDS].mean(axis=0)
    t = np.array([(u - w / 2) / f * Z, (v - h / 2) / f * Z, Z])
    return R, t


# ------------------------------------------------------------------ cube
CUBE_V = 0.5 * np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                         [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=np.float64)
CUBE_F = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
CUBE_N = np.array([[0, 0, -1], [0, 0, 1], [0, -1, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 0]], dtype=np.float64)
# Une couleur (BGR) par face : on voit tout de suite le cube tourner avec la paume.
# Face 1 (z = +1, verte) = celle qui regarde "vers le haut", loin de la paume.
FACE_COLORS = [(200, 120, 40), (90, 200, 90), (60, 60, 230), (60, 200, 230), (200, 90, 200), (230, 200, 60)]
LIGHT = unit(np.array([0.3, -0.5, -0.8]))            # direction VERS la lumière (repère caméra)


def project(P, f, cx, cy):
    P = np.atleast_2d(P)
    return np.column_stack([f * P[:, 0] / P[:, 2] + cx, f * P[:, 1] / P[:, 2] + cy])


FINGER_CHAINS = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]
PALM_HULL_IDS = [0, 1, 5, 9, 13, 17]


def hand_silhouette(px, Z, f, shape):
    """
    Silhouette approximative de la main, valeurs de 0 à 1, même taille que l'image.
    MediaPipe ne fournit pas de masque : on remplit la paume et on épaissit chaque os des doigts.
    """
    h, w = shape[:2]
    scale = f / Z                                      # mètres -> pixels à cette distance
    mask = np.zeros((h, w), np.uint8)
    pts = np.round(px).astype(np.int32)

    hull = cv2.convexHull(pts[PALM_HULL_IDS])
    cv2.fillConvexPoly(mask, hull, 255)
    cv2.polylines(mask, [hull], True, 255, max(1, int(0.02 * scale)))   # la peau dépasse un peu des points clés

    for k, chain in enumerate(FINGER_CHAINS):
        base = FINGER_THICKNESS * (1.25 if k == 0 else 1.0) * scale      # le pouce est plus épais
        for j, (a, b) in enumerate(zip(chain[:-1], chain[1:])):
            thick = max(2, int(base * (1.0 - 0.12 * j)))                 # le doigt s'affine vers le bout
            cv2.line(mask, tuple(pts[a]), tuple(pts[b]), 255, thick, cv2.LINE_AA)
            cv2.circle(mask, tuple(pts[b]), thick // 2, 255, -1, cv2.LINE_AA)  # bouts arrondis

    mask = cv2.GaussianBlur(mask, (0, 0), 1.5)         # bords un peu doux
    return mask.astype(np.float32) / 255.0


def draw_cube(img, R, t, size, f, hand_mask=None):
    """Dessine le cube. Si hand_mask est donné (0..1, taille de l'image), la main passe devant le cube."""
    h, w = img.shape[:2]
    center = t + R[:, 2] * (size / 2 + HOVER)        # le cube "pose" sur la paume, côté normale
    V = (CUBE_V * size) @ R.T + center                 # sommets : repère paume -> repère caméra
    if np.any(V[:, 2] < 0.05):
        return
    P = np.clip(project(V, f, w / 2, h / 2), -1e4, 1e4)
    N = CUBE_N @ R.T                                   # normales des faces, mêmes rotations

    visible = []
    for i, face in enumerate(CUBE_F):
        fc = V[list(face)].mean(axis=0)
        if np.dot(N[i], fc) < 0:                       # la face regarde la caméra (placée à l'origine)
            visible.append((fc[2], i))
    if not visible:
        return

    # On dessine le cube sur un calque + un masque "où le cube est dessiné", puis on mélange.
    layer = img.copy()
    cube_mask = np.zeros((h, w), np.uint8)
    for _, i in sorted(visible, reverse=True):         # du plus loin au plus proche
        shade = 0.45 + 0.55 * max(0.0, float(np.dot(N[i], LIGHT)))
        color = tuple(int(c * shade) for c in FACE_COLORS[i])
        poly = P[list(CUBE_F[i])].astype(np.int32)
        cv2.fillConvexPoly(layer, poly, color, cv2.LINE_AA)
        cv2.polylines(layer, [poly], True, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.fillConvexPoly(cube_mask, poly, 255, cv2.LINE_AA)
        cv2.polylines(cube_mask, [poly], True, 255, 1, cv2.LINE_AA)

    alpha = cube_mask.astype(np.float32) / 255.0
    if hand_mask is not None:
        alpha *= 1.0 - hand_mask                       # là où la main est présente, le cube disparaît

    # Mélange limité au rectangle qui entoure le cube (plus rapide que l'image entière)
    x0, y0 = np.clip(np.floor(P.min(axis=0)).astype(int) - 2, 0, [w, h])
    x1, y1 = np.clip(np.ceil(P.max(axis=0)).astype(int) + 3, 0, [w, h])
    if x1 <= x0 or y1 <= y0:
        return
    a = alpha[y0:y1, x0:x1, None]
    roi = img[y0:y1, x0:x1]
    roi[:] = (roi * (1.0 - a) + layer[y0:y1, x0:x1] * a).astype(np.uint8)


def draw_axes(img, R, t, f, length=0.06):
    """Debug : x rouge, y vert, z (normale, sort de la paume) bleu."""
    h, w = img.shape[:2]
    o = project(t[None, :], f, w / 2, h / 2)[0]
    for i, color in enumerate([(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
        e = project((t + R[:, i] * length)[None, :], f, w / 2, h / 2)[0]
        cv2.line(img, (int(o[0]), int(o[1])), (int(e[0]), int(e[1])), color, 2, cv2.LINE_AA)


# ------------------------------------------------------------------ affichage
CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
               (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
               (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


def draw_skeleton(img, px):
    pts = px.astype(np.int32)
    for a, b in CONNECTIONS:
        cv2.line(img, tuple(pts[a]), tuple(pts[b]), (255, 255, 255), 1, cv2.LINE_AA)
    for p in pts:
        cv2.circle(img, tuple(p), 3, (0, 200, 255), -1, cv2.LINE_AA)


def draw_hud(img, lines):
    y = 24
    for text in lines:
        cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22


# ------------------------------------------------------------------ programme principal
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true", help="force le CPU, ne tente pas le GPU")
    args = ap.parse_args()

    landmarker, delegate_used = build_landmarker(use_gpu=not args.cpu)
    print(f"Delegue utilise : {delegate_used}")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise RuntimeError("Impossible d'ouvrir la caméra 0")

    smooth_px, smooth_world = Ema(SMOOTHING), Ema(SMOOTHING)
    size = CUBE_SIZE
    show_skeleton, show_axes = True, True
    occlusion, show_mask = True, False
    t0 = time.perf_counter()
    last_ts = -1
    fps, t_prev = 0.0, time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # effet miroir
        h, w = frame.shape[:2]
        f = FOCAL_FACTOR * w

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int((time.perf_counter() - t0) * 1000)
        if ts <= last_ts:  # timestamps strictement croissants en mode VIDEO
            ts = last_ts + 1
        last_ts = ts

        result = landmarker.detect_for_video(mp_image, ts)

        info = "aucune main detectee"
        if result.hand_landmarks:
            lm = result.hand_landmarks[0]
            wl = result.hand_world_landmarks[0]
            label = result.handedness[0][0].category_name

            px = smooth_px(np.array([[p.x * w, p.y * h] for p in lm]))
            world = smooth_world(np.array([[p.x, p.y, p.z] for p in wl]))

            R, t = palm_pose(px, world, label, w, h, f)

            # Le cube est posé du côté de la normale. Si la normale pointe à l'opposé de la caméra
            # (produit scalaire normale . position > 0), le cube est derrière la main.
            behind = float(np.dot(R[:, 2], t)) > 0
            hand_mask = None
            if show_mask or (occlusion and behind):
                hand_mask = hand_silhouette(px, t[2], f, frame.shape)

            if show_skeleton:
                draw_skeleton(frame, px)
            if show_mask:  # debug : silhouette de la main en rouge
                m = hand_mask[..., None] * 0.45
                frame[:] = (frame * (1.0 - m) + np.array([0, 0, 255], np.float32) * m).astype(np.uint8)
            draw_cube(frame, R, t, size, f, hand_mask if (occlusion and behind) else None)
            if show_axes:
                draw_axes(frame, R, t, f)

            place = "derriere la main" if behind else "devant la main"
            info = f"{label} | distance {t[2] * 100:3.0f} cm | cube {place}"
        else:
            smooth_px.reset()
            smooth_world.reset()

        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
        t_prev = now

        draw_hud(frame, [
            f"FPS {fps:4.1f} | delegue : {delegate_used} | cube {size * 100:.1f} cm | occlusion {'on' if occlusion else 'off'}",
            info,
            "q quitter | s squelette | a axes | o occlusion | m masque | +/- taille",
        ])
        cv2.imshow("Cube dans la paume - etape 2", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        elif key == ord("s"):
            show_skeleton = not show_skeleton
        elif key == ord("a"):
            show_axes = not show_axes
        elif key == ord("o"):
            occlusion = not occlusion
        elif key == ord("m"):
            show_mask = not show_mask
        elif key in (ord("+"), ord("=")):
            size = min(size * 1.1, 0.15)
        elif key == ord("-"):
            size = max(size / 1.1, 0.01)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
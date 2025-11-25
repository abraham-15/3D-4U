# -*- coding: utf-8 -*-
"""
Cubo 3D morado controlado con manos
-----------------------------------
- Mano IZQUIERDA:
    * Controla el ZOOM (escala) del cubo usando la distancia entre
      pulgar (THUMB_TIP) e índice (INDEX_FINGER_TIP).
    * Sólo se toma en cuenta cuando REALMENTE está en la zona izquierda
      de la pantalla (wrist.x <= LEFT_HAND_MAX_X).
    * Cuando los dedos están muy juntos se fija en el zoom mínimo.
    * Cuando los dedos sobrepasan el límite de zoom máximo, se fija ahí.
    * En valores intermedios:
        - Si no detecta movimiento durante > 3s, el zoom se congela
          (evita cambios al retirar la mano).

- Mano DERECHA:
    * Gesto A: 2 dedos (índice + medio) juntos sobre el cubo:
        - Permite ARRASTRAR el cubo en X,Y.
        - El cubo se queda donde lo sueltas (dedos se separan o mano desaparece).
    * Gesto B: mano abierta (5 dedos) -> rotación del cubo:
        - Movimiento horizontal → rotación alrededor del eje Y.
        - Movimiento vertical → rotación alrededor del eje X.
        - Mover la mano de extremo a extremo ≈ giro de ~360°.
        - Al dejar de estar abierta o salir del cuadro, el cubo se queda en
          la última orientación.

- El cubo NO gira automáticamente: sólo se mueve/gira por interacción.
"""

import cv2
import mediapipe as mp
import numpy as np
import math
import time


# =====================================================
#  Render del cubo 3D
# =====================================================
class PurpleCubeRenderer:
    def __init__(self, base_size=1.0, distance=6.0, initial_scale=0.3):
        """
        Args:
            base_size: tamaño base de las coordenadas del cubo.
            distance: distancia en Z desde la cámara al centro del cubo.
            initial_scale: factor de escala inicial (cubo pequeño).
        """
        self.base_size = base_size
        self.distance = distance
        self.scale = initial_scale  # factor de escala (lo controla la mano izquierda)

        # Offset 2D para mover el cubo en la pantalla (lo controla la mano derecha)
        self.offset_2d = np.array([0.0, 0.0], dtype=np.float32)

        # Guardar últimos puntos proyectados y bbox para detección de "toque"
        self.last_pts_2d = None
        self.last_bbox = None  # (min_x, max_x, min_y, max_y)

        # Vértices de un cubo centrado en el origen con tamaño base_size
        s = self.base_size
        self.base_vertices = np.array([
            [-s, -s, -s],
            [-s, -s,  s],
            [-s,  s, -s],
            [-s,  s,  s],
            [ s, -s, -s],
            [ s, -s,  s],
            [ s,  s, -s],
            [ s,  s,  s],
        ], dtype=np.float32)

        # Caras como listas de índices de vértices (quads)
        self.faces = [
            [0, 1, 3, 2],  # izquierda
            [4, 5, 7, 6],  # derecha
            [0, 1, 5, 4],  # abajo
            [2, 3, 7, 6],  # arriba
            [0, 2, 6, 4],  # fondo
            [1, 3, 7, 5],  # frente
        ]

        # Tonos de morado (BGR)
        self.face_colors = [
            (128, 0, 128),   # morado medio
            (160, 32, 160),  # morado claro
            (96, 0, 128),    # morado oscuro
            (192, 64, 192),  # morado muy claro
            (80, 0, 96),     # morado muy oscuro
            (224, 96, 224),  # casi magenta
        ]

        # Ángulos de rotación (controlados por la mano derecha abierta)
        self.angle_x = 0.0
        self.angle_y = 0.0
        self.angle_z = 0.0

    # ---------- Rotaciones y proyección ----------
    def _rotation_matrix(self, ax, ay, az):
        """Matriz de rotación 3D Rz * Ry * Rx."""
        cx, sx = math.cos(ax), math.sin(ax)
        cy, sy = math.cos(ay), math.sin(ay)
        cz, sz = math.cos(az), math.sin(az)

        Rx = np.array([
            [1, 0, 0],
            [0, cx, -sx],
            [0, sx,  cx]
        ], dtype=np.float32)

        Ry = np.array([
            [ cy, 0, sy],
            [  0, 1,  0],
            [-sy, 0, cy]
        ], dtype=np.float32)

        Rz = np.array([
            [cz, -sz, 0],
            [sz,  cz, 0],
            [ 0,   0, 1]
        ], dtype=np.float32)

        return Rz @ Ry @ Rx

    def _project_points(self, points_3d, frame_shape, f=800.0):
        """
        Proyección perspectiva sencilla de puntos 3D al plano de la imagen.
        """
        h, w, _ = frame_shape
        cx, cy = w / 2.0, h / 2.0

        translated = points_3d.copy()
        translated[:, 2] += self.distance

        z = translated[:, 2].copy()
        z[z == 0] = 1e-6

        x = translated[:, 0]
        y = translated[:, 1]

        u = f * (x / z) + cx
        v = f * (-y / z) + cy

        points_2d = np.stack([u, v], axis=1).astype(np.float32)
        depths = z
        return points_2d, depths

    # ---------- Dibujo del cubo ----------
    def draw_cube(self, frame):
        """Dibuja el cubo con la escala, rotación y offset actuales sobre el frame."""
        # 1) Escalar y rotar vértices
        verts_scaled = self.base_vertices * self.scale
        R = self._rotation_matrix(self.angle_x, self.angle_y, self.angle_z)
        verts_rot = verts_scaled @ R.T

        # 2) Proyectar a 2D
        pts_2d, depths = self._project_points(verts_rot, frame.shape, f=800.0)

        # 3) Aplicar offset 2D (para mover el cubo con la mano derecha)
        pts_2d = pts_2d + self.offset_2d

        # Guardar para detección de toque
        self.last_pts_2d = pts_2d.copy()
        if pts_2d.size > 0:
            min_x = float(np.min(pts_2d[:, 0]))
            max_x = float(np.max(pts_2d[:, 0]))
            min_y = float(np.min(pts_2d[:, 1]))
            max_y = float(np.max(pts_2d[:, 1]))
            self.last_bbox = (min_x, max_x, min_y, max_y)
        else:
            self.last_bbox = None

        # 4) Ordenar caras por profundidad (pintar de atrás hacia adelante)
        face_depths = []
        for i, face in enumerate(self.faces):
            z_mean = depths[face].mean()
            face_depths.append((z_mean, i))
        face_depths.sort(reverse=True)

        # 5) Dibujar caras sólidas
        for _, face_idx in face_depths:
            face = self.faces[face_idx]
            color = self.face_colors[face_idx % len(self.face_colors)]

            pts_face = pts_2d[face].astype(np.int32).reshape((-1, 1, 2))

            cv2.fillConvexPoly(frame, pts_face, color)
            cv2.polylines(frame, [pts_face], isClosed=True,
                          color=(255, 255, 255), thickness=1)


# =====================================================
#  Main: cámara + mediapipe + cubo (zoom + drag + rotación)
# =====================================================
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return

    cube = PurpleCubeRenderer(base_size=1.0, distance=6.0, initial_scale=0.3)

    # Mediapipe Hands para detectar ambas manos
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    print("Cámara + cubo 3D.")
    print(" - Mano IZQUIERDA: zoom (distancia pulgar–índice, con congelado tras 3s).")
    print(" - Mano DERECHA :")
    print("      * índice+medio juntos sobre el cubo -> arrastre X,Y.")
    print("      * mano abierta (5 dedos) -> rotación del cubo en X,Y.")
    print("Presiona 'q' para salir.")

    # Parámetros para mapear distancia -> escala (zoom mano izquierda)
    MIN_PINCH = 0.02   # distancia mínima esperada (dedos casi juntos)
    MAX_PINCH = 0.40   # distancia máxima considerada (mano muy abierta)
    MIN_SCALE = 0.2    # cubo más pequeño
    MAX_SCALE = 1.0    # cubo más grande (no pasa de ~media pantalla)

    EDGE_DEAD_ZONE = 0.02  # zona muerta cerca de los extremos min/max

    # Congelado de zoom en zona intermedia
    FREEZE_SECS = 3.0          # tiempo sin movimiento para congelar zoom
    MOVEMENT_EPS = 0.003       # cambio mínimo en pinch_dist para “movimiento”
    LARGE_MOVEMENT_EPS = 0.03  # cambio grande para “descongelar”

    # IMPORTANTE: zona X válida (normalizada) para considerar la mano izquierda.
    # Sólo si el WRIST.x <= LEFT_HAND_MAX_X, permitimos zoom.
    LEFT_HAND_MAX_X = 0.45

    last_pinch_dist_mid = None
    last_zoom_change_time_mid = None
    zoom_frozen = False

    # Mano derecha: parámetros y estado de arrastre
    GRIP_DIST_MAX = 0.10  # distancia máx (normalizada) entre índice y medio para “agarre”
    dragging = False
    drag_start_avg = None      # (x, y) promedio de índice+medio al iniciar drag
    drag_start_offset = None   # copia de cube.offset_2d al iniciar drag

    # Mano derecha: estado de rotación con mano abierta
    rotating = False
    rotate_start_pos = None      # (x_norm, y_norm) del punto de referencia (muñeca)
    rotate_start_angles = None   # (angle_x, angle_y) al inicio de la rotación

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # espejo

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        left_seen = False
        right_seen = False

        if result.multi_hand_landmarks and result.multi_handedness:
            for hand_landmarks, handedness in zip(
                result.multi_hand_landmarks, result.multi_handedness
            ):
                label = handedness.classification[0].label  # 'Left' o 'Right'
                lm = mp_hands.HandLandmark

                wrist = hand_landmarks.landmark[lm.WRIST]
                wrist_x = wrist.x  # coordenada normalizada (0 izquierda, 1 derecha)

                # ===== Mano IZQUIERDA (zoom) SOLO en zona izquierda =====
                if label == 'Left' and wrist_x <= LEFT_HAND_MAX_X:
                    left_seen = True
                    now = time.time()

                    thumb_tip = hand_landmarks.landmark[lm.THUMB_TIP]        # 4
                    index_tip = hand_landmarks.landmark[lm.INDEX_FINGER_TIP] # 8

                    thumb_norm = (thumb_tip.x, thumb_tip.y)
                    index_norm = (index_tip.x, index_tip.y)

                    # Distancia normalizada pulgar–índice
                    pinch_dist = float(
                        np.linalg.norm(np.array(thumb_norm) - np.array(index_norm))
                    )

                    # Coordenadas en píxeles para debug/visual
                    tx = int(thumb_tip.x * frame.shape[1])
                    ty = int(thumb_tip.y * frame.shape[0])
                    ix = int(index_tip.x * frame.shape[1])
                    iy = int(index_tip.y * frame.shape[0])

                    # Puntos de referencia en la imagen
                    cv2.circle(frame, (tx, ty), 6, (0, 255, 0), -1)
                    cv2.circle(frame, (ix, iy), 6, (0, 255, 0), -1)

                    # Límites de zona intermedia
                    MID_LOW = MIN_PINCH + EDGE_DEAD_ZONE
                    MID_HIGH = MAX_PINCH - EDGE_DEAD_ZONE

                    if pinch_dist <= MID_LOW:
                        # Zona de dedos casi totalmente juntos -> zoom mínimo, sin más interacción
                        cube.scale = MIN_SCALE
                        last_pinch_dist_mid = None
                        last_zoom_change_time_mid = None
                        zoom_frozen = False
                    elif pinch_dist >= MID_HIGH:
                        # Zona de dedos muy separados -> zoom máximo, sin más interacción
                        cube.scale = MAX_SCALE
                        last_pinch_dist_mid = None
                        last_zoom_change_time_mid = None
                        zoom_frozen = False
                    else:
                        # Zona intermedia: se controla el zoom con congelado tras 3s
                        if last_pinch_dist_mid is None:
                            last_pinch_dist_mid = pinch_dist
                            last_zoom_change_time_mid = now
                            zoom_frozen = False

                        delta = abs(pinch_dist - last_pinch_dist_mid)

                        # Si ya estaba congelado, sólo reactivamos con movimiento grande
                        if zoom_frozen:
                            if delta > LARGE_MOVEMENT_EPS:
                                zoom_frozen = False
                                last_zoom_change_time_mid = now
                            else:
                                last_pinch_dist_mid = pinch_dist
                        # Si no está congelado, actualizamos si hay movimiento suficiente
                        if not zoom_frozen:
                            if delta > MOVEMENT_EPS:
                                t = (pinch_dist - MIN_PINCH) / (MAX_PINCH - MIN_PINCH)
                                t = max(0.0, min(1.0, t))  # clamp

                                cube.scale = MIN_SCALE + t * (MAX_SCALE - MIN_SCALE)
                                last_pinch_dist_mid = pinch_dist
                                last_zoom_change_time_mid = now
                            else:
                                # Sin movimiento significativo, checar tiempo para congelar
                                if (last_zoom_change_time_mid is not None and
                                        now - last_zoom_change_time_mid > FREEZE_SECS):
                                    zoom_frozen = True
                                last_pinch_dist_mid = pinch_dist

                    # Debug en consola
                    print(
                        f"[LEFT ZOOM] pinch_dist={pinch_dist:.4f} "
                        f"scale={cube.scale:.3f} "
                        f"frozen={zoom_frozen} "
                        f"wrist_x={wrist_x:.3f} "
                        f"thumb_px=({tx:4d},{ty:4d}) index_px=({ix:4d},{iy:4d})"
                    )

                # ===== Mano DERECHA: DRAG (2 dedos) y ROTACIÓN (mano abierta) =====
                elif label == 'Right':
                    right_seen = True

                    index_tip = hand_landmarks.landmark[lm.INDEX_FINGER_TIP]   # 8
                    middle_tip = hand_landmarks.landmark[lm.MIDDLE_FINGER_TIP] # 12
                    ring_tip = hand_landmarks.landmark[lm.RING_FINGER_TIP]     # 16
                    pinky_tip = hand_landmarks.landmark[lm.PINKY_TIP]          # 20

                    index_pip = hand_landmarks.landmark[lm.INDEX_FINGER_PIP]
                    middle_pip = hand_landmarks.landmark[lm.MIDDLE_FINGER_PIP]
                    ring_pip = hand_landmarks.landmark[lm.RING_FINGER_PIP]
                    pinky_pip = hand_landmarks.landmark[lm.PINKY_PIP]

                    # Coordenadas normalizadas
                    index_norm = (index_tip.x, index_tip.y)
                    middle_norm = (middle_tip.x, middle_tip.y)

                    # Coordenadas en píxeles
                    ix = int(index_tip.x * frame.shape[1])
                    iy = int(index_tip.y * frame.shape[0])
                    mx = int(middle_tip.x * frame.shape[1])
                    my = int(middle_tip.y * frame.shape[0])

                    # Puntos azules en índice y medio (indicadores)
                    cv2.circle(frame, (ix, iy), 8, (255, 0, 0), -1)
                    cv2.circle(frame, (mx, my), 8, (255, 0, 0), -1)

                    # ---- Detección de mano abierta (4 dedos extendidos) ----
                    def finger_up(tip, pip):
                        # y menor = más arriba en la imagen
                        return tip.y < pip.y

                    index_up = finger_up(index_tip, index_pip)
                    middle_up = finger_up(middle_tip, middle_pip)
                    ring_up = finger_up(ring_tip, ring_pip)
                    pinky_up = finger_up(pinky_tip, pinky_pip)

                    count_up = sum([index_up, middle_up, ring_up, pinky_up])
                    open_hand = (count_up == 4)  # mano abierta (ignoramos pulgar)

                    # Distancia entre índice y medio (para saber si están "juntos" agarrando)
                    pair_dist = float(
                        np.linalg.norm(np.array(index_norm) - np.array(middle_norm))
                    )
                    grip_close = pair_dist < GRIP_DIST_MAX

                    # Promedio de los dos dedos (punto de agarre para drag)
                    avg_x = 0.5 * (ix + mx)
                    avg_y = 0.5 * (iy + my)

                    # ===== GESTO B: MANO ABIERTA -> ROTACIÓN =====
                    if open_hand:
                        # Si estábamos arrastrando, cerramos el drag al entrar en rotación
                        if dragging:
                            print("[RIGHT DRAG] Fin de arrastre (cambio a rotación)")
                            dragging = False
                            drag_start_avg = None
                            drag_start_offset = None

                        wrist_norm = (wrist.x, wrist.y)

                        if not rotating:
                            rotating = True
                            rotate_start_pos = wrist_norm
                            rotate_start_angles = (cube.angle_x, cube.angle_y)
                            print("[RIGHT ROTATE] Inicio de rotación con mano abierta")
                        else:
                            dx = wrist_norm[0] - rotate_start_pos[0]
                            dy = wrist_norm[1] - rotate_start_pos[1]

                            # Mano a la derecha -> giro a la derecha
                            # Mano hacia abajo  -> giro hacia abajo
                            cube.angle_y = rotate_start_angles[1] - dx * 2.0 * math.pi
                            cube.angle_x = rotate_start_angles[0] - dy * 2.0 * math.pi

                            print(
                                f"[RIGHT ROTATE] wrist_norm=({wrist_norm[0]:.3f},{wrist_norm[1]:.3f}) "
                                f"angles=({cube.angle_x:.3f},{cube.angle_y:.3f})"
                            )

                    # ===== Gesto A: índice+medio -> DRAG EN X,Y =====
                    else:
                        # Si ya no hay mano abierta, cortar rotación si estaba activa
                        if rotating:
                            print("[RIGHT ROTATE] Fin de rotación (mano dejó de estar abierta)")
                        rotating = False
                        rotate_start_pos = None
                        rotate_start_angles = None

                        if not dragging:
                            # Comprobar si ambos dedos están dentro del cubo al INICIAR el drag
                            inside = False
                            if cube.last_bbox is not None:
                                min_x, max_x, min_y, max_y = cube.last_bbox
                                if (min_x <= ix <= max_x and min_x <= mx <= max_x and
                                    min_y <= iy <= max_y and min_y <= my <= max_y):
                                    inside = True

                            if inside and grip_close:
                                dragging = True
                                drag_start_avg = (avg_x, avg_y)
                                drag_start_offset = cube.offset_2d.copy()
                                print("[RIGHT DRAG] Inicio de arrastre sobre el cubo")
                        else:
                            # Ya estamos arrastrando: seguir mientras los dedos sigan juntos
                            if not grip_close:
                                print("[RIGHT DRAG] Fin de arrastre (dedos separados)")
                                dragging = False
                                drag_start_avg = None
                                drag_start_offset = None
                            else:
                                dx = avg_x - drag_start_avg[0]
                                dy = avg_y - drag_start_avg[1]
                                cube.offset_2d = drag_start_offset + np.array(
                                    [dx, dy], dtype=np.float32
                                )

                                print(
                                    f"[RIGHT DRAG] avg=({avg_x:5.1f},{avg_y:5.1f}) "
                                    f"offset=({cube.offset_2d[0]:5.1f},{cube.offset_2d[1]:5.1f})"
                                )

        # Si en este frame no vimos la mano izquierda en su zona válida, reseteamos su estado interno
        if not left_seen:
            last_pinch_dist_mid = None
            last_zoom_change_time_mid = None
            zoom_frozen = False

        # Si en este frame no vimos la mano derecha, detenemos arrastre y rotación
        if not right_seen:
            if dragging:
                print("[RIGHT DRAG] Fin de arrastre (mano derecha no detectada)")
            if rotating:
                print("[RIGHT ROTATE] Fin de rotación (mano derecha no detectada)")
            dragging = False
            drag_start_avg = None
            drag_start_offset = None
            rotating = False
            rotate_start_pos = None
            rotate_start_angles = None

        # ---------- DIBUJAR CUBO ----------
        cube.draw_cube(frame)

        # Mostrar texto de escala actual
        cv2.putText(
            frame,
            f"Scale: {cube.scale:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.imshow("3D-4U - Cubo 3D (zoom+drag+rotate)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

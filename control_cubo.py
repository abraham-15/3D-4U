# control_cubo.py
# -*- coding: utf-8 -*-
"""
Control del cubo 3D morado con manos (MediaPipe)
-------------------------------------------------
- Mano IZQUIERDA:
    * Pinch (pulgar–índice) -> ZOOM del cubo (en cualquier parte).
    * PAUSA: sólo índice extendido, los demás dedos doblados,
      índice colocado SOBRE el cubo y pulgar lejos del índice.
      Mientras este gesto esté activo, se detiene zoom, drag y rotación.

- Mano DERECHA:
    * Menú (en header superior):
        - Botón "CuboUnico" en la esquina superior izquierda.
        - Solo índice extendido, índice SOBRE el botón durante ≥ 3s
          => activa el modelo 3D del cubo (CuboUnico).
        - Cuando el cubo está activo, el botón "CuboUnico" desaparece
          y aparece un botón "Regresar" en la esquina superior derecha.
        - Solo índice extendido, índice SOBRE "Regresar" durante ≥ 3s
          => desactiva el cubo y vuelve al menú.
    * Cubo (cuando CuboUnico está activo y no hay pausa):
        - índice + medio juntos sobre el cubo -> ARRASTRAR el cubo en X,Y.
        - mano abierta (4 dedos extendidos) -> ROTACIÓN del cubo (X,Y).
"""

import cv2
import mediapipe as mp
import numpy as np
import math
import time

from cube_renderer import PurpleCubeRenderer


def finger_up(tip, pip):
    """Determina si un dedo está 'levantado' (y menor = más arriba en la imagen)."""
    return tip.y < pip.y


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return

    cube = PurpleCubeRenderer(base_size=1.0, distance=6.0, initial_scale=0.3)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    print("Cámara + cubo 3D.")
    print(" - Mano IZQUIERDA: zoom (pinch) + pausa (solo índice sobre el cubo).")
    print(" - Mano DERECHA :")
    print("      * Header: botón 'CuboUnico' (3s) -> activa cubo.")
    print("      * Header: botón 'Regresar' (3s) -> vuelve al menú.")
    print("      * Cubo activo: índice+medio -> drag, mano abierta -> rotación.")
    print("Presiona 'q' para salir.")

    # Parámetros zoom mano izquierda
    MIN_PINCH = 0.02
    MAX_PINCH = 0.40
    MIN_SCALE = 0.2
    MAX_SCALE = 1.0

    EDGE_DEAD_ZONE = 0.02

    FREEZE_SECS = 3.0
    MOVEMENT_EPS = 0.003
    LARGE_MOVEMENT_EPS = 0.03

    # Umbral para considerar que el pulgar está lo bastante lejos del índice
    # y NO es un gesto de pinch (para la pausa).
    PAUSE_THUMB_MIN_DIST = 0.08  # ajustable

    last_pinch_dist_mid = None
    last_zoom_change_time_mid = None
    zoom_frozen = False

    # Mano derecha: drag
    GRIP_DIST_MAX = 0.10
    dragging = False
    drag_start_avg = None
    drag_start_offset = None

    # Mano derecha: rotación
    rotating = False
    rotate_start_pos = None
    rotate_start_angles = None

    # Menú / header
    HEADER_H = 70               # altura del header en píxeles
    MENU_HOLD_SECS = 3.0        # tiempo de "hold" para activar botón

    menu_hover_start_cubo = None
    back_hover_start = None
    active_model = None  # "CuboUnico" cuando se active el cubo

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        now = time.time()

        h, w, _ = frame.shape

        # Coordenadas de botones dentro del header
        MENU_BTN_X = 10
        MENU_BTN_Y = 10
        MENU_BTN_W = 170
        MENU_BTN_H = HEADER_H - 20  # deja margen arriba/abajo

        BACK_BTN_W = 150
        BACK_BTN_H = HEADER_H - 20
        BACK_BTN_X = w - BACK_BTN_W - 10
        BACK_BTN_Y = 10

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        left_seen = False
        right_seen = False
        interaction_paused = False

        # ---------- 1) Detectar GESTO DE PAUSA con mano izquierda ----------
        if result.multi_hand_landmarks and result.multi_handedness:
            for hand_landmarks, handedness in zip(
                result.multi_hand_landmarks, result.multi_handedness
            ):
                label = handedness.classification[0].label
                lm = mp_hands.HandLandmark

                if label == 'Left':
                    index_tip = hand_landmarks.landmark[lm.INDEX_FINGER_TIP]
                    index_pip = hand_landmarks.landmark[lm.INDEX_FINGER_PIP]
                    middle_tip = hand_landmarks.landmark[lm.MIDDLE_FINGER_TIP]
                    middle_pip = hand_landmarks.landmark[lm.MIDDLE_FINGER_PIP]
                    ring_tip = hand_landmarks.landmark[lm.RING_FINGER_TIP]
                    ring_pip = hand_landmarks.landmark[lm.RING_FINGER_PIP]
                    pinky_tip = hand_landmarks.landmark[lm.PINKY_TIP]
                    pinky_pip = hand_landmarks.landmark[lm.PINKY_PIP]
                    thumb_tip = hand_landmarks.landmark[lm.THUMB_TIP]

                    index_up = finger_up(index_tip, index_pip)
                    middle_up = finger_up(middle_tip, middle_pip)
                    ring_up = finger_up(ring_tip, ring_pip)
                    pinky_up = finger_up(pinky_tip, pinky_pip)

                    index_only_up = index_up and not (middle_up or ring_up or pinky_up)

                    thumb_norm = np.array([thumb_tip.x, thumb_tip.y])
                    index_norm = np.array([index_tip.x, index_tip.y])
                    thumb_index_dist = float(
                        np.linalg.norm(thumb_norm - index_norm)
                    )
                    thumb_far = thumb_index_dist > PAUSE_THUMB_MIN_DIST

                    ix = int(index_tip.x * frame.shape[1])
                    iy = int(index_tip.y * frame.shape[0])
                    index_on_cube = False
                    if cube.last_bbox is not None:
                        min_x, max_x, min_y, max_y = cube.last_bbox
                        if min_x <= ix <= max_x and min_y <= iy <= max_y:
                            index_on_cube = True

                    if index_only_up and index_on_cube and thumb_far:
                        interaction_paused = True
                        break

        if interaction_paused:
            if dragging:
                print("[PAUSA] Fin de arrastre (pausa activada).")
            if rotating:
                print("[PAUSA] Fin de rotación (pausa activada).")
            dragging = False
            drag_start_avg = None
            drag_start_offset = None
            rotating = False
            rotate_start_pos = None
            rotate_start_angles = None

        # ---------- 2) Procesar interacciones (menú / zoom / drag / rotación) ----------
        if result.multi_hand_landmarks and result.multi_handedness:
            for hand_landmarks, handedness in zip(
                result.multi_hand_landmarks, result.multi_handedness
            ):
                label = handedness.classification[0].label
                lm = mp_hands.HandLandmark

                wrist = hand_landmarks.landmark[lm.WRIST]
                wrist_x = wrist.x

                # ===== Mano IZQUIERDA: ZOOM (si no está pausado) =====
                if label == 'Left':
                    left_seen = True

                    if interaction_paused:
                        continue

                    thumb_tip = hand_landmarks.landmark[lm.THUMB_TIP]
                    index_tip = hand_landmarks.landmark[lm.INDEX_FINGER_TIP]

                    thumb_norm = (thumb_tip.x, thumb_tip.y)
                    index_norm = (index_tip.x, index_tip.y)

                    pinch_dist = float(
                        np.linalg.norm(np.array(thumb_norm) - np.array(index_norm))
                    )

                    tx = int(thumb_tip.x * frame.shape[1])
                    ty = int(thumb_tip.y * frame.shape[0])
                    ix = int(index_tip.x * frame.shape[1])
                    iy = int(index_tip.y * frame.shape[0])

                    cv2.circle(frame, (tx, ty), 6, (0, 255, 0), -1)
                    cv2.circle(frame, (ix, iy), 6, (0, 255, 0), -1)

                    MID_LOW = MIN_PINCH + EDGE_DEAD_ZONE
                    MID_HIGH = MAX_PINCH - EDGE_DEAD_ZONE

                    if pinch_dist <= MID_LOW:
                        cube.scale = MIN_SCALE
                        last_pinch_dist_mid = None
                        last_zoom_change_time_mid = None
                        zoom_frozen = False
                    elif pinch_dist >= MID_HIGH:
                        cube.scale = MAX_SCALE
                        last_pinch_dist_mid = None
                        last_zoom_change_time_mid = None
                        zoom_frozen = False
                    else:
                        if last_pinch_dist_mid is None:
                            last_pinch_dist_mid = pinch_dist
                            last_zoom_change_time_mid = now
                            zoom_frozen = False

                        delta = abs(pinch_dist - last_pinch_dist_mid)

                        if zoom_frozen:
                            if delta > LARGE_MOVEMENT_EPS:
                                zoom_frozen = False
                                last_zoom_change_time_mid = now
                            else:
                                last_pinch_dist_mid = pinch_dist

                        if not zoom_frozen:
                            if delta > MOVEMENT_EPS:
                                t = (pinch_dist - MIN_PINCH) / (MAX_PINCH - MIN_PINCH)
                                t = max(0.0, min(1.0, t))

                                cube.scale = MIN_SCALE + t * (MAX_SCALE - MIN_SCALE)
                                last_pinch_dist_mid = pinch_dist
                                last_zoom_change_time_mid = now
                            else:
                                if (last_zoom_change_time_mid is not None and
                                        now - last_zoom_change_time_mid > FREEZE_SECS):
                                    zoom_frozen = True
                                last_pinch_dist_mid = pinch_dist

                    print(
                        f"[LEFT ZOOM] pinch_dist={pinch_dist:.4f} "
                        f"scale={cube.scale:.3f} "
                        f"frozen={zoom_frozen} "
                        f"wrist_x={wrist_x:.3f} "
                        f"thumb_px=({tx:4d},{ty:4d}) index_px=({ix:4d},{iy:4d})"
                    )

                # ===== Mano DERECHA: MENÚ + (si cubo activo y no pausado) DRAG/ROTACIÓN =====
                elif label == 'Right':
                    right_seen = True

                    # Coordenadas y estados de dedos
                    index_tip = hand_landmarks.landmark[lm.INDEX_FINGER_TIP]
                    middle_tip = hand_landmarks.landmark[lm.MIDDLE_FINGER_TIP]
                    ring_tip = hand_landmarks.landmark[lm.RING_FINGER_TIP]
                    pinky_tip = hand_landmarks.landmark[lm.PINKY_TIP]

                    index_pip = hand_landmarks.landmark[lm.INDEX_FINGER_PIP]
                    middle_pip = hand_landmarks.landmark[lm.MIDDLE_FINGER_PIP]
                    ring_pip = hand_landmarks.landmark[lm.RING_FINGER_PIP]
                    pinky_pip = hand_landmarks.landmark[lm.PINKY_PIP]

                    index_up = finger_up(index_tip, index_pip)
                    middle_up = finger_up(middle_tip, middle_pip)
                    ring_up = finger_up(ring_tip, ring_pip)
                    pinky_up = finger_up(pinky_tip, pinky_pip)

                    index_only_up_right = index_up and not (middle_up or ring_up or pinky_up)

                    ix = int(index_tip.x * frame.shape[1])
                    iy = int(index_tip.y * frame.shape[0])

                    # ---------- MENÚ EN HEADER ----------
                    inside_menu_btn = (
                        MENU_BTN_X <= ix <= MENU_BTN_X + MENU_BTN_W and
                        MENU_BTN_Y <= iy <= MENU_BTN_Y + MENU_BTN_H
                    )

                    inside_back_btn = False
                    if active_model == "CuboUnico":
                        inside_back_btn = (
                            BACK_BTN_X <= ix <= BACK_BTN_X + BACK_BTN_W and
                            BACK_BTN_Y <= iy <= BACK_BTN_Y + BACK_BTN_H
                        )

                    # Lógica de selección de botones (solo índice extendido)
                    if index_only_up_right and inside_menu_btn and active_model != "CuboUnico":
                        # Hover sobre "CuboUnico"
                        if menu_hover_start_cubo is None:
                            menu_hover_start_cubo = now
                        else:
                            elapsed = now - menu_hover_start_cubo
                            if elapsed >= MENU_HOLD_SECS and active_model != "CuboUnico":
                                active_model = "CuboUnico"
                                menu_hover_start_cubo = None
                                back_hover_start = None
                                print("[MENU] Modelo 'CuboUnico' activado.")
                    elif index_only_up_right and inside_back_btn and active_model == "CuboUnico":
                        # Hover sobre "Regresar"
                        if back_hover_start is None:
                            back_hover_start = now
                        else:
                            elapsed = now - back_hover_start
                            if elapsed >= MENU_HOLD_SECS and active_model == "CuboUnico":
                                active_model = None
                                back_hover_start = None
                                menu_hover_start_cubo = None
                                print("[MENU] Regresando al menú (cubo desactivado).")
                    else:
                        # No está sosteniendo ningún botón con índice
                        menu_hover_start_cubo = None if not inside_menu_btn else menu_hover_start_cubo
                        back_hover_start = None if not inside_back_btn else back_hover_start
                        # Nota: si quieres que se “reinicie” inmediato al salir, esto está bien.

                    # Si la interacción está pausada O no hay cubo, no hacemos drag/rotación
                    if interaction_paused or active_model != "CuboUnico":
                        continue

                    # ---------- A partir de aquí: gestos para el cubo (solo si activo) ----------
                    index_norm = (index_tip.x, index_tip.y)
                    middle_norm = (middle_tip.x, middle_tip.y)

                    mx = int(middle_tip.x * frame.shape[1])
                    my = int(middle_tip.y * frame.shape[0])

                    cv2.circle(frame, (ix, iy), 8, (255, 0, 0), -1)
                    cv2.circle(frame, (mx, my), 8, (255, 0, 0), -1)

                    count_up = sum([index_up, middle_up, ring_up, pinky_up])
                    open_hand = (count_up == 4)

                    pair_dist = float(
                        np.linalg.norm(np.array(index_norm) - np.array(middle_norm))
                    )
                    grip_close = pair_dist < GRIP_DIST_MAX

                    avg_x = 0.5 * (ix + mx)
                    avg_y = 0.5 * (iy + my)

                    # ---- Rotación con mano abierta ----
                    if open_hand:
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

                            cube.angle_y = rotate_start_angles[1] - dx * 2.0 * math.pi
                            cube.angle_x = rotate_start_angles[0] - dy * 2.0 * math.pi

                            print(
                                f"[RIGHT ROTATE] wrist_norm=({wrist_norm[0]:.3f},{wrist_norm[1]:.3f}) "
                                f"angles=({cube.angle_x:.3f},{cube.angle_y:.3f})"
                            )

                    # ---- Drag con índice+medio juntos ----
                    else:
                        if rotating:
                            print("[RIGHT ROTATE] Fin de rotación (mano dejó de estar abierta)")
                        rotating = False
                        rotate_start_pos = None
                        rotate_start_angles = None

                        if not dragging:
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

        # Si no vimos mano izquierda, reseteamos estado del zoom intermedio
        if not left_seen:
            last_pinch_dist_mid = None
            last_zoom_change_time_mid = None
            zoom_frozen = False

        # Si no vimos mano derecha, terminamos drag/rotación
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

        # ---------- DIBUJAR CUBO (solo si el modelo está activo) ----------
        if active_model == "CuboUnico":
            cube.draw_cube(frame)

        # ---------- DIBUJAR HEADER Y BOTONES (por encima de cámara/modelo) ----------
        # Header opaco en la parte superior (para asegurar visibilidad)
        cv2.rectangle(
            frame,
            (0, 0),
            (w, HEADER_H),
            (40, 40, 40),  # gris oscuro sólido
            -1
        )

        # Botón de selección de modelo (solo cuando NO hay modelo activo)
        if active_model != "CuboUnico":
            # Fondo del botón
            cv2.rectangle(
                frame,
                (MENU_BTN_X, MENU_BTN_Y),
                (MENU_BTN_X + MENU_BTN_W, MENU_BTN_Y + MENU_BTN_H),
                (70, 70, 200),  # azul-grisáceo sólido
                -1
            )
            # Borde
            cv2.rectangle(
                frame,
                (MENU_BTN_X, MENU_BTN_Y),
                (MENU_BTN_X + MENU_BTN_W, MENU_BTN_Y + MENU_BTN_H),
                (255, 255, 255),
                2
            )

            # Barra de progreso de selección (si se está sosteniendo el índice)
            if menu_hover_start_cubo is not None:
                elapsed = now - menu_hover_start_cubo
                progress = max(0.0, min(1.0, elapsed / MENU_HOLD_SECS))
                fill_w = int(MENU_BTN_W * progress)
                cv2.rectangle(
                    frame,
                    (MENU_BTN_X, MENU_BTN_Y),
                    (MENU_BTN_X + fill_w, MENU_BTN_Y + MENU_BTN_H),
                    (100, 255, 100),  # verde claro para feedback
                    -1
                )

            # Texto
            cv2.putText(
                frame,
                "CuboUnico",
                (MENU_BTN_X + 10, MENU_BTN_Y + int(MENU_BTN_H * 0.7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        # Botón de regreso (solo cuando hay modelo activo)
        else:
            cv2.rectangle(
                frame,
                (BACK_BTN_X, BACK_BTN_Y),
                (BACK_BTN_X + BACK_BTN_W, BACK_BTN_Y + BACK_BTN_H),
                (200, 70, 70),  # rojizo para indicar 'volver'
                -1
            )
            cv2.rectangle(
                frame,
                (BACK_BTN_X, BACK_BTN_Y),
                (BACK_BTN_X + BACK_BTN_W, BACK_BTN_Y + BACK_BTN_H),
                (255, 255, 255),
                2
            )

            if back_hover_start is not None:
                elapsed = now - back_hover_start
                progress = max(0.0, min(1.0, elapsed / MENU_HOLD_SECS))
                fill_w = int(BACK_BTN_W * progress)
                cv2.rectangle(
                    frame,
                    (BACK_BTN_X, BACK_BTN_Y),
                    (BACK_BTN_X + fill_w, BACK_BTN_Y + BACK_BTN_H),
                    (255, 200, 100),  # naranja claro
                    -1
                )

            cv2.putText(
                frame,
                "Regresar",
                (BACK_BTN_X + 10, BACK_BTN_Y + int(BACK_BTN_H * 0.7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        # ---------- TEXTOS INFORMATIVOS (debajo del header) ----------
        cv2.putText(
            frame,
            f"Scale: {cube.scale:.2f}",
            (10, HEADER_H + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        if interaction_paused and active_model == "CuboUnico":
            cv2.putText(
                frame,
                "PAUSA: indice izq sobre el cubo",
                (10, HEADER_H + 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

        cv2.imshow("3D-4U - Header + Menú + Cubo 3D", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

# control_cubo.py
# -*- coding: utf-8 -*-
"""
Control de modelos 3D con manos (MediaPipe)
-------------------------------------------
Modelos:
- "CuboUnico": cubo morado único (PurpleCubeRenderer).
- "4 Cubos"  : cubo compuesto de 4 piezas (CompositeCubes).

Header:
- Sin modelo activo:
    [CuboUnico]  [4 Cubos]
- Con modelo activo:
    [Regresar] (en la esquina superior derecha).

Selección de botones:
- Mano DERECHA, SOLO el índice extendido (medio, anular y meñique doblados).
- Mantener la punta del índice sobre el botón ≥ 3s.

Gestos comunes:
- Mano IZQUIERDA:
    * Pinch (pulgar–índice) -> ZOOM del modelo activo.
    * PAUSA: solo índice extendido, demás dedos doblados,
      índice sobre el modelo (cubo único o uno de los 4 cubos) y pulgar lejos.
      Mientras esté activa, se detiene la interacción global
      y, en el modelo de 4 cubos, se habilita la selección fina de piezas.

- Mano DERECHA:
    * Menú: selección de modelo / regresar (solo índice extendido sobre botones).
    * Modelo "CuboUnico" (si no hay pausa):
        - índice + medio juntos sobre el cubo -> ARRASTRAR el cubo (offset 2D).
        - mano abierta (4 dedos) -> ROTACIÓN global del cubo.
    * Modelo "4 Cubos":
        - Sin pausa:
            - mano abierta (4 dedos):
                · si todas las piezas están adjuntas -> rotación global.
                · si hay piezas separadas -> rotación local del cubo más cercano.
            - índice + medio juntos -> arrastre del grupo (solo piezas adjuntas).
        - Con pausa:
            - índice + medio juntos sobre un cubo -> seleccionar y mover SOLO esa pieza
              (CompositeCubes.move_piece), respetando colisiones.
"""

import cv2
import mediapipe as mp
import numpy as np
import math
import time

from cube_renderer import PurpleCubeRenderer
from cubo_4_piezas import CompositeCubes


def finger_up(tip, pip):
    """Determina si un dedo está 'levantado' (y menor = más arriba en la imagen)."""
    return tip.y < pip.y


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return

    # Modelos
    cube = PurpleCubeRenderer(base_size=1.0, distance=6.0, initial_scale=0.3)
    composite = CompositeCubes(initial_scale=0.3)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    print("Cámara + modelos 3D.")
    print("Header:")
    print("  - Botón 'CuboUnico'")
    print("  - Botón '4 Cubos'")
    print("Gestos:")
    print("  - Selección de botones: mano derecha, solo índice extendido, 3s sobre el botón.")
    print("  - Zoom (modelos activos): mano izquierda pinch (pulgar+índice).")
    print("  - Pausa (modelos activos): mano izquierda solo índice sobre el modelo.")
    print("  - CuboUnico:")
    print("      * derecha índice+medio -> arrastrar cubo.")
    print("      * derecha mano abierta -> rotar cubo.")
    print("  - 4 Cubos:")
    print("      * sin pausa: mano abierta -> rotación global / de pieza cercana.")
    print("      * sin pausa: índice+medio -> arrastrar grupo (piezas adjuntas).")
    print("      * con pausa: índice+medio sobre un cubo -> mover solo esa pieza.")
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

    # Pulgar lo bastante lejos para NO considerarse pinch (para pausa)
    PAUSE_THUMB_MIN_DIST = 0.08

    last_pinch_dist_mid = None
    last_zoom_change_time_mid = None
    zoom_frozen = False

    # Mano derecha: estados para CuboUnico
    single_dragging = False
    single_drag_start_avg = None
    single_drag_start_offset = None

    single_rotating = False
    single_rotate_start_pos = None
    single_rotate_start_angles = None

    # Mano derecha: estados para 4 cubos (grupo)
    comp_group_dragging = False
    comp_group_last_avg = None

    comp_rotating = False
    comp_rotate_mode = None  # "global" o "piece"
    comp_rotate_start_pos = None
    comp_rotate_start_angles_global = None
    comp_rotate_piece_index = None
    comp_rotate_piece_local_start = None

    # Mano derecha: estados para 4 cubos (pieza individual en pausa)
    comp_piece_dragging = False
    comp_piece_drag_index = None
    comp_piece_drag_last_avg = None

    # Menú / header
    HEADER_H = 70
    MENU_HOLD_SECS = 2.0

    # Timers para botones
    menu_hover_start_cubo = None        # botón "CuboUnico"
    menu_hover_start_4c = None          # botón "4 Cubos"
    back_hover_start = None             # botón "Regresar"

    # Modelo activo: None / "CuboUnico" / "Cubo4"
    active_model = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        now = time.time()
        h, w, _ = frame.shape

        # Coordenadas header y botones
        HEADER_Y0 = 0
        HEADER_Y1 = HEADER_H

        # Botón CuboUnico
        CU_BTN_X = 10
        CU_BTN_Y = HEADER_Y0 + 10
        CU_BTN_W = 170
        CU_BTN_H = HEADER_H - 20

        # Botón 4 Cubos (a la derecha del anterior)
        FOUR_BTN_X = CU_BTN_X + CU_BTN_W + 10
        FOUR_BTN_Y = CU_BTN_Y
        FOUR_BTN_W = 170
        FOUR_BTN_H = CU_BTN_H

        # Botón Regresar (esquina superior derecha)
        BACK_BTN_W = 150
        BACK_BTN_H = HEADER_H - 20
        BACK_BTN_X = w - BACK_BTN_W - 10
        BACK_BTN_Y = HEADER_Y0 + 10

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        left_seen = False
        right_seen = False
        interaction_paused = False

        # =====================================================
        # 1) GESTO DE PAUSA (mano izquierda, solo índice sobre el modelo)
        # =====================================================
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

                    ix = int(index_tip.x * w)
                    iy = int(index_tip.y * h)

                    index_on_model = False
                    if active_model == "CuboUnico" and cube.last_bbox is not None:
                        min_x, max_x, min_y, max_y = cube.last_bbox
                        if min_x <= ix <= max_x and min_y <= iy <= max_y:
                            index_on_model = True
                    elif active_model == "Cubo4":
                        # índice sobre cualquier pieza
                        for piece in composite.pieces:
                            if piece.last_bbox is None:
                                continue
                            min_x, max_x, min_y, max_y = piece.last_bbox
                            if min_x <= ix <= max_x and min_y <= iy <= max_y:
                                index_on_model = True
                                break

                    if index_only_up and index_on_model and thumb_far:
                        interaction_paused = True
                        break

        # Si se activa pausa, se cortan los movimientos globales actuales
        if interaction_paused:
            if single_dragging:
                print("[PAUSA] Fin arrastre cubo único.")
            if single_rotating:
                print("[PAUSA] Fin rotación cubo único.")
            single_dragging = False
            single_drag_start_avg = None
            single_drag_start_offset = None
            single_rotating = False
            single_rotate_start_pos = None
            single_rotate_start_angles = None

            if comp_group_dragging:
                print("[PAUSA] Fin arrastre grupo 4 cubos.")
            comp_group_dragging = False
            comp_group_last_avg = None

            if comp_rotating:
                print("[PAUSA] Fin rotación 4 cubos.")
            comp_rotating = False
            comp_rotate_mode = None
            comp_rotate_start_pos = None
            comp_rotate_start_angles_global = None
            comp_rotate_piece_index = None
            comp_rotate_piece_local_start = None

        # =====================================================
        # 2) INTERACCIONES (zoom / menú / modelos)
        # =====================================================
        if result.multi_hand_landmarks and result.multi_handedness:
            for hand_landmarks, handedness in zip(
                result.multi_hand_landmarks, result.multi_handedness
            ):
                label = handedness.classification[0].label
                lm = mp_hands.HandLandmark
                wrist = hand_landmarks.landmark[lm.WRIST]

                # ----------------------------
                # Mano IZQUIERDA: ZOOM
                # ----------------------------
                if label == 'Left':
                    left_seen = True

                    # Si no hay modelo activo, no hacemos nada (favor de activar uno)
                    if active_model is None or interaction_paused:
                        continue

                    thumb_tip = hand_landmarks.landmark[lm.THUMB_TIP]
                    index_tip = hand_landmarks.landmark[lm.INDEX_FINGER_TIP]

                    thumb_norm = (thumb_tip.x, thumb_tip.y)
                    index_norm = (index_tip.x, index_tip.y)

                    pinch_dist = float(
                        np.linalg.norm(np.array(thumb_norm) - np.array(index_norm))
                    )

                    tx = int(thumb_tip.x * w)
                    ty = int(thumb_tip.y * h)
                    ix = int(index_tip.x * w)
                    iy = int(index_tip.y * h)

                    cv2.circle(frame, (tx, ty), 6, (0, 255, 0), -1)
                    cv2.circle(frame, (ix, iy), 6, (0, 255, 0), -1)

                    MID_LOW = MIN_PINCH + EDGE_DEAD_ZONE
                    MID_HIGH = MAX_PINCH - EDGE_DEAD_ZONE

                    if pinch_dist <= MID_LOW:
                        new_scale = MIN_SCALE
                        last_pinch_dist_mid = None
                        last_zoom_change_time_mid = None
                        zoom_frozen = False
                    elif pinch_dist >= MID_HIGH:
                        new_scale = MAX_SCALE
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
                                new_scale = MIN_SCALE + t * (MAX_SCALE - MIN_SCALE)
                                last_pinch_dist_mid = pinch_dist
                                last_zoom_change_time_mid = now
                            else:
                                if (last_zoom_change_time_mid is not None and
                                        now - last_zoom_change_time_mid > FREEZE_SECS):
                                    zoom_frozen = True
                                last_pinch_dist_mid = pinch_dist
                                # mantener última escala
                                if active_model == "CuboUnico":
                                    new_scale = cube.scale
                                else:
                                    new_scale = composite.scale

                    # Aplicar escala al modelo activo
                    if active_model == "CuboUnico":
                        cube.scale = new_scale
                        scale_print = cube.scale
                    else:
                        composite.scale = new_scale
                        scale_print = composite.scale

                    print(
                        f"[LEFT ZOOM] model={active_model} pinch_dist={pinch_dist:.4f} "
                        f"scale={scale_print:.3f} frozen={zoom_frozen}"
                    )

                # ----------------------------
                # Mano DERECHA: MENÚ + MODELOS
                # ----------------------------
                elif label == 'Right':
                    right_seen = True

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

                    ix = int(index_tip.x * w)
                    iy = int(index_tip.y * h)

                    # Para gestos sobre modelos
                    index_norm = (index_tip.x, index_tip.y)
                    middle_norm = (middle_tip.x, middle_tip.y)
                    mx = int(middle_tip.x * w)
                    my = int(middle_tip.y * h)

                    # ---------------- MENU (header) ----------------
                    inside_cu_btn = (
                        CU_BTN_X <= ix <= CU_BTN_X + CU_BTN_W and
                        CU_BTN_Y <= iy <= CU_BTN_Y + CU_BTN_H
                    )
                    inside_4c_btn = (
                        FOUR_BTN_X <= ix <= FOUR_BTN_X + FOUR_BTN_W and
                        FOUR_BTN_Y <= iy <= FOUR_BTN_Y + FOUR_BTN_H
                    )
                    inside_back_btn = (
                        BACK_BTN_X <= ix <= BACK_BTN_X + BACK_BTN_W and
                        BACK_BTN_Y <= iy <= BACK_BTN_Y + BACK_BTN_H
                    ) if active_model is not None else False

                    if index_only_up_right:
                        # Sin modelo activo: selección de CuboUnico o 4 Cubos
                        if active_model is None:
                            if inside_cu_btn:
                                # hovering CuboUnico
                                if menu_hover_start_cubo is None:
                                    menu_hover_start_cubo = now
                                else:
                                    elapsed = now - menu_hover_start_cubo
                                    if elapsed >= MENU_HOLD_SECS:
                                        active_model = "CuboUnico"
                                        cube.offset_2d[:] = 0.0
                                        menu_hover_start_cubo = None
                                        menu_hover_start_4c = None
                                        back_hover_start = None
                                        print("[MENU] Modelo 'CuboUnico' activado.")
                            else:
                                menu_hover_start_cubo = None

                            if inside_4c_btn:
                                if menu_hover_start_4c is None:
                                    menu_hover_start_4c = now
                                else:
                                    elapsed = now - menu_hover_start_4c
                                    if elapsed >= MENU_HOLD_SECS:
                                        active_model = "Cubo4"
                                        composite.group_offset_2d[:] = 0.0
                                        menu_hover_start_cubo = None
                                        menu_hover_start_4c = None
                                        back_hover_start = None
                                        print("[MENU] Modelo '4 Cubos' activado.")
                            else:
                                menu_hover_start_4c = None

                        # Con modelo activo: botón Regresar
                        else:
                            menu_hover_start_cubo = None
                            menu_hover_start_4c = None
                            if inside_back_btn:
                                if back_hover_start is None:
                                    back_hover_start = now
                                else:
                                    elapsed = now - back_hover_start
                                    if elapsed >= MENU_HOLD_SECS:
                                        print("[MENU] Regresando al menú (modelo desactivado).")
                                        active_model = None
                                        back_hover_start = None
                                        # Reset estados de interacción
                                        single_dragging = False
                                        single_rotating = False
                                        comp_group_dragging = False
                                        comp_rotating = False
                                        comp_piece_dragging = False
                            else:
                                back_hover_start = None
                    else:
                        # No hay gesto de menú activo
                        if active_model is None:
                            menu_hover_start_cubo = None
                            menu_hover_start_4c = None
                        back_hover_start = None

                    # Si no hay modelo activo, o estamos solo en menú, no seguimos con gestos de modelo
                    if active_model is None:
                        continue

                    # ---------------- GESTOS PARA MODELOS ----------------
                    # Mano abierta / índice+medio juntos
                    count_up = sum([index_up, middle_up, ring_up, pinky_up])
                    open_hand = (count_up == 4)

                    pair_dist = float(
                        np.linalg.norm(np.array(index_norm) - np.array(middle_norm))
                    )
                    GRIP_DIST_MAX = 0.10
                    grip_close = pair_dist < GRIP_DIST_MAX

                    avg_x = 0.5 * (ix + mx)
                    avg_y = 0.5 * (iy + my)

                    # ===== Modelo: CuboUnico =====
                    if active_model == "CuboUnico":
                        if interaction_paused:
                            # En pausa no hay gestos de movimiento para el cubo único
                            continue

                        # Dibujar puntos de referencia para índices/medio
                        cv2.circle(frame, (ix, iy), 8, (255, 0, 0), -1)
                        cv2.circle(frame, (mx, my), 8, (255, 0, 0), -1)

                        if open_hand:
                            # Rotación global cubo único
                            if single_dragging:
                                print("[RIGHT DRAG] Fin de arrastre cubo único (cambio a rotación).")
                                single_dragging = False
                                single_drag_start_avg = None
                                single_drag_start_offset = None

                            wrist_norm = (wrist.x, wrist.y)
                            if not single_rotating:
                                single_rotating = True
                                single_rotate_start_pos = wrist_norm
                                single_rotate_start_angles = (cube.angle_x, cube.angle_y)
                                print("[RIGHT ROTATE] Inicio de rotación cubo único.")
                            else:
                                dx = wrist_norm[0] - single_rotate_start_pos[0]
                                dy = wrist_norm[1] - single_rotate_start_pos[1]
                                cube.angle_y = single_rotate_start_angles[1] - dx * 2.0 * math.pi
                                cube.angle_x = single_rotate_start_angles[0] - dy * 2.0 * math.pi

                        else:
                            # Rotación termina si deja de estar la mano abierta
                            if single_rotating:
                                print("[RIGHT ROTATE] Fin de rotación cubo único.")
                            single_rotating = False
                            single_rotate_start_pos = None
                            single_rotate_start_angles = None

                            # Arrastre con índice+medio
                            if not single_dragging:
                                inside = False
                                if cube.last_bbox is not None:
                                    min_x, max_x, min_y, max_y = cube.last_bbox
                                    if (min_x <= ix <= max_x and min_x <= mx <= max_x and
                                        min_y <= iy <= max_y and min_y <= my <= max_y):
                                        inside = True
                                if inside and grip_close:
                                    single_dragging = True
                                    single_drag_start_avg = (avg_x, avg_y)
                                    single_drag_start_offset = cube.offset_2d.copy()
                                    print("[RIGHT DRAG] Inicio arrastre cubo único.")
                            else:
                                if not grip_close:
                                    print("[RIGHT DRAG] Fin arrastre cubo único (dedos separados).")
                                    single_dragging = False
                                    single_drag_start_avg = None
                                    single_drag_start_offset = None
                                else:
                                    dx = avg_x - single_drag_start_avg[0]
                                    dy = avg_y - single_drag_start_avg[1]
                                    cube.offset_2d = single_drag_start_offset + np.array(
                                        [dx, dy], dtype=np.float32
                                    )

                    # ===== Modelo: 4 Cubos (CompositeCubes) =====
                    elif active_model == "Cubo4":
                        # Dibujo de puntos
                        cv2.circle(frame, (ix, iy), 8, (255, 0, 0), -1)
                        cv2.circle(frame, (mx, my), 8, (255, 0, 0), -1)

                        # --- Pausa: solo gestos finos (mover pieza) ---
                        if interaction_paused:
                            # No rotación global ni arrastre de grupo
                            comp_group_dragging = False
                            if comp_rotating:
                                print("[RIGHT ROTATE 4C] Fin rotación (pausa).")
                                comp_rotating = False
                                comp_rotate_mode = None
                                comp_rotate_start_pos = None
                                comp_rotate_start_angles_global = None
                                comp_rotate_piece_index = None
                                comp_rotate_piece_local_start = None

                            # Mover pieza individual con índice+medio
                            if not grip_close:
                                if comp_piece_dragging:
                                    print(f"[4C] Fin movimiento pieza {comp_piece_drag_index}.")
                                comp_piece_dragging = False
                                comp_piece_drag_index = None
                                comp_piece_drag_last_avg = None
                            else:
                                if not comp_piece_dragging:
                                    # Buscar pieza bajo el gesto
                                    chosen_idx = None
                                    for j, piece in enumerate(composite.pieces):
                                        if piece.last_bbox is None:
                                            continue
                                        min_x, max_x, min_y, max_y = piece.last_bbox
                                        if (min_x <= ix <= max_x and min_y <= iy <= max_y and
                                            min_x <= mx <= max_x and min_y <= my <= max_y):
                                            chosen_idx = j
                                            break
                                    if chosen_idx is not None:
                                        comp_piece_dragging = True
                                        comp_piece_drag_index = chosen_idx
                                        comp_piece_drag_last_avg = (avg_x, avg_y)
                                        print(f"[4C] Inicio movimiento pieza {chosen_idx} (modo pausa).")
                                else:
                                    # Movimiento incremental de la pieza seleccionada
                                    dx = avg_x - comp_piece_drag_last_avg[0]
                                    dy = avg_y - comp_piece_drag_last_avg[1]
                                    composite.move_piece(comp_piece_drag_index, dx, dy)
                                    comp_piece_drag_last_avg = (avg_x, avg_y)
                            continue  # no seguir con rotación/arrastre global

                        # --- Sin pausa: arrastre de grupo y rotación ---
                        comp_piece_dragging = False
                        comp_piece_drag_index = None
                        comp_piece_drag_last_avg = None

                        # Arrastre de grupo con índice+medio
                        if not open_hand and grip_close:
                            # Mientras no estemos rotando
                            if not comp_group_dragging:
                                # Comprobar si estamos sobre alguna pieza adjunta
                                inside_any = False
                                for piece in composite.pieces:
                                    if piece.last_bbox is None or not piece.attached:
                                        continue
                                    min_x, max_x, min_y, max_y = piece.last_bbox
                                    if (min_x <= ix <= max_x and min_y <= iy <= max_y and
                                        min_x <= mx <= max_x and min_y <= my <= max_y):
                                        inside_any = True
                                        break
                                if inside_any:
                                    comp_group_dragging = True
                                    comp_group_last_avg = (avg_x, avg_y)
                                    print("[4C] Inicio arrastre grupo (piezas adjuntas).")
                            else:
                                dx = avg_x - comp_group_last_avg[0]
                                dy = avg_y - comp_group_last_avg[1]
                                composite.move_group(dx, dy)
                                comp_group_last_avg = (avg_x, avg_y)
                        else:
                            if comp_group_dragging and (not grip_close):
                                print("[4C] Fin arrastre grupo (dedos separados o mano abierta).")
                            comp_group_dragging = False
                            comp_group_last_avg = None

                        # Rotación con mano abierta
                        if open_hand:
                            if comp_group_dragging:
                                print("[4C] Fin arrastre grupo (cambio a rotación).")
                                comp_group_dragging = False
                                comp_group_last_avg = None

                            wrist_norm = (wrist.x, wrist.y)

                            all_attached = all(p.attached for p in composite.pieces)

                            if not comp_rotating:
                                comp_rotating = True
                                comp_rotate_start_pos = wrist_norm

                                if all_attached:
                                    comp_rotate_mode = "global"
                                    comp_rotate_start_angles_global = (
                                        composite.angle_x, composite.angle_y
                                    )
                                    print("[4C ROTATE] Inicio rotación global (todas adjuntas).")
                                else:
                                    # Rotación de la pieza más cercana
                                    # Usamos el índice como referencia de proximidad
                                    ref_x, ref_y = ix, iy
                                    best_j = None
                                    best_dist = None
                                    for j, piece in enumerate(composite.pieces):
                                        if piece.last_bbox is None:
                                            continue
                                        min_x, max_x, min_y, max_y = piece.last_bbox
                                        cx = 0.5 * (min_x + max_x)
                                        cy = 0.5 * (min_y + max_y)
                                        d = (ref_x - cx) ** 2 + (ref_y - cy) ** 2
                                        if best_dist is None or d < best_dist:
                                            best_dist = d
                                            best_j = j
                                    if best_j is not None:
                                        comp_rotate_mode = "piece"
                                        comp_rotate_piece_index = best_j
                                        piece = composite.pieces[best_j]
                                        comp_rotate_piece_local_start = piece.local_angles.copy()
                                        print(f"[4C ROTATE] Inicio rotación pieza {best_j}.")
                                    else:
                                        comp_rotating = False
                                        comp_rotate_mode = None
                            else:
                                dx = wrist_norm[0] - comp_rotate_start_pos[0]
                                dy = wrist_norm[1] - comp_rotate_start_pos[1]
                                if comp_rotate_mode == "global":
                                    composite.angle_y = (
                                        comp_rotate_start_angles_global[1] - dx * 2.0 * math.pi
                                    )
                                    composite.angle_x = (
                                        comp_rotate_start_angles_global[0] - dy * 2.0 * math.pi
                                    )
                                elif comp_rotate_mode == "piece":
                                    idxp = comp_rotate_piece_index
                                    if 0 <= idxp < len(composite.pieces):
                                        piece = composite.pieces[idxp]
                                        piece.local_angles[1] = (
                                            comp_rotate_piece_local_start[1] - dx * 2.0 * math.pi
                                        )
                                        piece.local_angles[0] = (
                                            comp_rotate_piece_local_start[0] - dy * 2.0 * math.pi
                                        )
                        else:
                            if comp_rotating:
                                print("[4C ROTATE] Fin rotación.")
                            comp_rotating = False
                            comp_rotate_mode = None
                            comp_rotate_start_pos = None
                            comp_rotate_start_angles_global = None
                            comp_rotate_piece_index = None
                            comp_rotate_piece_local_start = None

        # Si no se vio mano izquierda, reseteamos estado del zoom intermedio
        if not left_seen:
            last_pinch_dist_mid = None
            last_zoom_change_time_mid = None
            zoom_frozen = False

        # Si no se vio mano derecha, reseteamos interacciones derechas
        if not right_seen:
            if single_dragging:
                print("[RIGHT DRAG] Fin arrastre cubo único (sin mano derecha).")
            if single_rotating:
                print("[RIGHT ROTATE] Fin rotación cubo único (sin mano derecha).")
            single_dragging = False
            single_drag_start_avg = None
            single_drag_start_offset = None
            single_rotating = False
            single_rotate_start_pos = None
            single_rotate_start_angles = None

            if comp_group_dragging:
                print("[4C] Fin arrastre grupo (sin mano derecha).")
            comp_group_dragging = False
            comp_group_last_avg = None

            if comp_rotating:
                print("[4C ROTATE] Fin rotación (sin mano derecha).")
            comp_rotating = False
            comp_rotate_mode = None
            comp_rotate_start_pos = None
            comp_rotate_start_angles_global = None
            comp_rotate_piece_index = None
            comp_rotate_piece_local_start = None

            if comp_piece_dragging:
                print(f"[4C] Fin movimiento pieza {comp_piece_drag_index} (sin mano derecha).")
            comp_piece_dragging = False
            comp_piece_drag_index = None
            comp_piece_drag_last_avg = None

        # =====================================================
        # 3) DIBUJAR MODELO ACTIVO
        # =====================================================
        if active_model == "CuboUnico":
            cube.draw_cube(frame)
        elif active_model == "Cubo4":
            composite.draw(frame)

        # =====================================================
        # 4) DIBUJAR HEADER Y BOTONES
        # =====================================================
        # Header opaco
        cv2.rectangle(
            frame,
            (0, HEADER_Y0),
            (w, HEADER_Y1),
            (40, 40, 40),
            -1
        )

        # Menú sin modelo activo: dos botones
        if active_model is None:
            # Botón CuboUnico
            cv2.rectangle(
                frame,
                (CU_BTN_X, CU_BTN_Y),
                (CU_BTN_X + CU_BTN_W, CU_BTN_Y + CU_BTN_H),
                (70, 70, 200),
                -1
            )
            cv2.rectangle(
                frame,
                (CU_BTN_X, CU_BTN_Y),
                (CU_BTN_X + CU_BTN_W, CU_BTN_Y + CU_BTN_H),
                (255, 255, 255),
                2
            )
            if menu_hover_start_cubo is not None:
                elapsed = now - menu_hover_start_cubo
                progress = max(0.0, min(1.0, elapsed / MENU_HOLD_SECS))
                fill_w = int(CU_BTN_W * progress)
                cv2.rectangle(
                    frame,
                    (CU_BTN_X, CU_BTN_Y),
                    (CU_BTN_X + fill_w, CU_BTN_Y + CU_BTN_H),
                    (100, 255, 100),
                    -1
                )
            cv2.putText(
                frame,
                "CuboUnico",
                (CU_BTN_X + 10, CU_BTN_Y + int(CU_BTN_H * 0.7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            # Botón 4 Cubos
            cv2.rectangle(
                frame,
                (FOUR_BTN_X, FOUR_BTN_Y),
                (FOUR_BTN_X + FOUR_BTN_W, FOUR_BTN_Y + FOUR_BTN_H),
                (70, 200, 70),
                -1
            )
            cv2.rectangle(
                frame,
                (FOUR_BTN_X, FOUR_BTN_Y),
                (FOUR_BTN_X + FOUR_BTN_W, FOUR_BTN_Y + FOUR_BTN_H),
                (255, 255, 255),
                2
            )
            if menu_hover_start_4c is not None:
                elapsed = now - menu_hover_start_4c
                progress = max(0.0, min(1.0, elapsed / MENU_HOLD_SECS))
                fill_w = int(FOUR_BTN_W * progress)
                cv2.rectangle(
                    frame,
                    (FOUR_BTN_X, FOUR_BTN_Y),
                    (FOUR_BTN_X + fill_w, FOUR_BTN_Y + FOUR_BTN_H),
                    (100, 255, 100),
                    -1
                )
            cv2.putText(
                frame,
                "4 Cubos",
                (FOUR_BTN_X + 10, FOUR_BTN_Y + int(FOUR_BTN_H * 0.7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
        else:
            # Botón Regresar
            cv2.rectangle(
                frame,
                (BACK_BTN_X, BACK_BTN_Y),
                (BACK_BTN_X + BACK_BTN_W, BACK_BTN_Y + BACK_BTN_H),
                (200, 70, 70),
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
                    (255, 200, 100),
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

        # Texto de escala
        if active_model == "CuboUnico":
            current_scale = cube.scale
        elif active_model == "Cubo4":
            current_scale = composite.scale
        else:
            current_scale = 0.0

        cv2.putText(
            frame,
            f"Scale: {current_scale:.2f}",
            (10, HEADER_H + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        if interaction_paused and active_model is not None:
            cv2.putText(
                frame,
                "PAUSA: indice izq sobre el modelo",
                (10, HEADER_H + 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

        cv2.imshow("3D-4U - Menu + Modelos 3D", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

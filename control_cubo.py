# -*- coding: utf-8 -*-
"""
control_cubo.py

Modelos:
- CuboUnico  -> PurpleCubeRenderer (cube_renderer.py)
- 4 Cubos    -> CompositeCubes (cubo_4_piezas.py)

Gestos principales
------------------
MANO IZQUIERDA
- Activar zoom (una sola vez por objetivo):
    * Índice extendido.
    * Pulgar visible.
    * Distancia índice–pulgar MUY pequeña (pinch fuerte) sobre el modelo.
- Zoom (solo si está activado para el objetivo actual):
    * Índice extendido.
    * Pulgar visible y a una distancia razonable.
    * Distancia índice–pulgar controla el zoom:
        - Muy juntos  -> MIN_SCALE (no sigue bajando).
        - Muy separados -> MAX_SCALE (no sigue subiendo).
        - Zona intermedia -> zoom suave con congelado de 3 s si no hay movimiento.
- Stop:
    * Solo índice extendido (middle, ring, pinky doblados).
    * Índice sobre el modelo.
    * Pulgar muy lejos del índice o casi no visible.
    * Mientras está en STOP se detiene la interacción global (drag/rotación).

MANO DERECHA
- Menú:
    * Solo índice extendido, 3 s sobre botón -> activa modelo o regresar.
- 4 Cubos:
    * Solo índice extendido, 2 s sobre cubo:
        - Si está pegado al bloque -> se separa + queda seleccionado.
        - Si ya estaba separado   -> sólo se selecciona.
    * Gesto de grupo:
        - Mano abierta -> mano cerrada (0 dedos extendidos) en < 0.8 s:
            -> selecciona el bloque completo como objetivo de interacción,
               aunque haya piezas separadas.
    * Si NO hay cubo seleccionado (grupo objetivo):
        - Índice+medio juntos -> arrastre del bloque.
        - Mano abierta        -> rotación global bloque.
    * Si HAY cubo seleccionado y separado:
        - Índice+medio juntos -> arrastre de esa pieza.
        - Mano abierta        -> rotación de esa pieza.
- CuboUnico:
    * Índice+medio juntos sobre cubo -> arrastre.
    * Mano abierta                   -> rotación global.
"""

import cv2
import mediapipe as mp
import numpy as np
import math
import time

from cube_renderer import PurpleCubeRenderer
from cubo_4_piezas import CompositeCubes


# ============================================================
# Utilidades
# ============================================================

def finger_up(tip, pip):
    """True si la punta está por encima del PIP (menor y en la imagen)."""
    return tip.y < pip.y


# ============================================================
# Main
# ============================================================

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return

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
    print("Header: CuboUnico / 4 Cubos (selección con mano derecha, solo índice, 3s).")
    print("Zoom: mano izquierda (pinch pulgar+índice, activable por pinch fuerte).")
    print("Stop: mano izquierda, solo índice extendido, pulgar lejos/no visible, índice sobre el modelo.")
    print("4 Cubos: piezas seleccionables (2s) y gesto de grupo (mano abierta->cerrada).")
    print("Presiona 'q' para salir.")

    # ---------- Constantes zoom ----------
    MIN_PINCH = 0.02     # dedos casi juntos
    MAX_PINCH = 0.40     # mano muy abierta
    MIN_SCALE = 0.2
    MAX_SCALE = 1.0
    EDGE_DEAD_ZONE = 0.02

    FREEZE_SECS = 3.0
    MOVEMENT_EPS = 0.003
    LARGE_MOVEMENT_EPS = 0.03

    # Umbrales mano izquierda
    ACTIVATION_PINCH_DIST = 0.045  # pinch muy fuerte para activar zoom
    PINCH_VALID_MAX_DIST = 0.35    # máximo para considerar pinch para zoom
    PAUSE_THUMB_DIST = 0.32        # pulgar lejos -> candidato a STOP
    MIN_THUMB_VIS = 0.3            # visibilidad mínima para considerar pulgar “visible”

    # Selección 2s mano derecha (4 Cubos)
    SELECT_HOLD_SECS = 2.0

    # Header / menú
    HEADER_H = 70
    MENU_HOLD_SECS = 3.0

    # Mano derecha: grip índice+medio
    GRIP_DIST_MAX = 0.10

    # Gesto grupo: open -> closed en ventana de tiempo
    RIGHT_TOGGLE_WINDOW = 0.8

    # -------- Estados zoom mano izquierda --------
    last_pinch_dist_mid = None
    last_zoom_change_time_mid = None
    zoom_frozen = False

    zoom_enabled = False   # zoom habilitado para el objetivo actual

    # -------- Estados CuboUnico --------
    single_dragging = False
    single_drag_start_avg = None
    single_drag_start_offset = None

    single_rotating = False
    single_rot_start_pos = None
    single_rot_start_angles = None

    # -------- Estados 4 Cubos --------
    comp_group_dragging = False
    comp_group_drag_last_avg = None  # para bloque

    comp_piece_dragging = False
    comp_piece_drag_last_avg = None  # para pieza

    comp_rotating = False
    comp_rot_target = None        # "block" / "piece"
    comp_rot_piece_index = None
    comp_rot_start_pos = None
    comp_rot_block_start = None   # (ax, ay)
    comp_rot_piece_start = None   # copia de local_angles

    comp_selected_index = None    # None -> bloque; int -> pieza seleccionada

    comp_hold_index = None        # candidato para 2s
    comp_hold_start = None

    # Menú / header
    menu_hover_start_cubo = None
    menu_hover_start_4c = None
    back_hover_start = None

    active_model = None  # None / "CuboUnico" / "Cubo4"

    # Estado mano derecha para gesto de grupo
    right_state_prev = "other"
    right_state_change_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        now = time.time()
        h, w, _ = frame.shape

        HEADER_Y0, HEADER_Y1 = 0, HEADER_H

        # Botones
        CU_BTN_X = 10
        CU_BTN_Y = HEADER_Y0 + 10
        CU_BTN_W = 170
        CU_BTN_H = HEADER_H - 20
        FOUR_BTN_X = CU_BTN_X + CU_BTN_W + 10
        FOUR_BTN_Y = CU_BTN_Y
        FOUR_BTN_W = 170
        FOUR_BTN_H = CU_BTN_H
        BACK_BTN_W = 150
        BACK_BTN_H = HEADER_H - 20
        BACK_BTN_X = w - BACK_BTN_W - 10
        BACK_BTN_Y = HEADER_Y0 + 10

        CU_BTN_rect = (CU_BTN_X, CU_BTN_Y, CU_BTN_X + CU_BTN_W, CU_BTN_Y + CU_BTN_H)
        FOUR_BTN_rect = (FOUR_BTN_X, FOUR_BTN_Y, FOUR_BTN_X + FOUR_BTN_W, FOUR_BTN_Y + FOUR_BTN_H)
        BACK_BTN_rect = (BACK_BTN_X, BACK_BTN_Y, BACK_BTN_X + BACK_BTN_W, BACK_BTN_Y + BACK_BTN_H)

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        # Extraer una mano izquierda y una derecha (si existen)
        left_lm = None
        right_lm = None
        if result.multi_hand_landmarks and result.multi_handedness:
            for lmks, handedness in zip(
                result.multi_hand_landmarks, result.multi_handedness
            ):
                label = handedness.classification[0].label
                if label == 'Left' and left_lm is None:
                    left_lm = (lmks, mp_hands.HandLandmark)
                elif label == 'Right' and right_lm is None:
                    right_lm = (lmks, mp_hands.HandLandmark)

        interaction_paused = False
        scale_value = 0.0

        # =====================================================
        # 1) MANO IZQUIERDA: STOP + ACTIVACIÓN DE ZOOM + ZOOM
        # =====================================================
        if left_lm is not None and active_model is not None:
            hand_landmarks, lm = left_lm

            index_tip = hand_landmarks.landmark[lm.INDEX_FINGER_TIP]
            index_pip = hand_landmarks.landmark[lm.INDEX_FINGER_PIP]
            middle_tip = hand_landmarks.landmark[lm.MIDDLE_FINGER_TIP]
            middle_pip = hand_landmarks.landmark[lm.MIDDLE_FINGER_PIP]
            ring_tip = hand_landmarks.landmark[lm.RING_FINGER_TIP]
            ring_pip = hand_landmarks.landmark[lm.RING_FINGER_PIP]
            pinky_tip = hand_landmarks.landmark[lm.PINKY_TIP]
            pinky_pip = hand_landmarks.landmark[lm.PINKY_PIP]
            thumb_tip = hand_landmarks.landmark[lm.THUMB_TIP]

            # Dedos arriba
            index_up_l = finger_up(index_tip, index_pip)
            middle_up_l = finger_up(middle_tip, middle_pip)
            ring_up_l = finger_up(ring_tip, ring_pip)
            pinky_up_l = finger_up(pinky_tip, pinky_pip)

            thumb_visibility = getattr(thumb_tip, "visibility", 1.0)

            # Solo índice extendido (para STOP)
            index_only_l = index_up_l and not (middle_up_l or ring_up_l or pinky_up_l)

            # Distancia pulgar–índice
            thumb_norm = np.array([thumb_tip.x, thumb_tip.y], dtype=np.float32)
            index_norm = np.array([index_tip.x, index_tip.y], dtype=np.float32)
            thumb_index_dist = float(np.linalg.norm(thumb_norm - index_norm))

            thumb_far_for_stop = (thumb_index_dist > PAUSE_THUMB_DIST or
                                  thumb_visibility < MIN_THUMB_VIS)
            pinch_valid = (thumb_visibility >= MIN_THUMB_VIS and
                           thumb_index_dist < PINCH_VALID_MAX_DIST)

            ix_l = int(index_tip.x * w)
            iy_l = int(index_tip.y * h)

            # ¿Índice sobre algún modelo?
            index_on_model = False
            if active_model == "CuboUnico" and getattr(cube, "last_bbox", None) is not None:
                mnx, mxx, mny, mxy = cube.last_bbox
                if mnx <= ix_l <= mxx and mny <= iy_l <= mxy:
                    index_on_model = True
            elif active_model == "Cubo4":
                for p in composite.pieces:
                    if p.last_bbox is None:
                        continue
                    mnx, mxx, mny, mxy = p.last_bbox
                    if mnx <= ix_l <= mxx and mny <= iy_l <= mxy:
                        index_on_model = True
                        break

            # ---------- STOP / PAUSA ----------
            if index_only_l and index_on_model and thumb_far_for_stop:
                if not interaction_paused:
                    print("[PAUSA] Activada por mano izquierda.")
                interaction_paused = True
                # Al pausar no tocamos zoom_enabled, solo limpiamos buffers de movimiento.
                last_pinch_dist_mid = None
                last_zoom_change_time_mid = None
                zoom_frozen = False
            else:
                # ---------- ACTIVACIÓN DE ZOOM ----------
                if (not zoom_enabled and pinch_valid and index_up_l and index_on_model and
                        thumb_index_dist <= ACTIVATION_PINCH_DIST):
                    zoom_enabled = True
                    last_pinch_dist_mid = None
                    last_zoom_change_time_mid = None
                    zoom_frozen = False
                    print("[LEFT ZOOM] Zoom habilitado para objetivo actual.")

                # ---------- ZOOM (solo si está habilitado) ----------
                if zoom_enabled and pinch_valid and index_up_l:
                    pinch_dist = thumb_index_dist

                    # Debug visual de pinch
                    tx = int(thumb_tip.x * w)
                    ty = int(thumb_tip.y * h)
                    cv2.circle(frame, (tx, ty), 6, (0, 255, 0), -1)
                    cv2.circle(frame, (ix_l, iy_l), 6, (0, 255, 0), -1)

                    MID_LOW = MIN_PINCH + EDGE_DEAD_ZONE
                    MID_HIGH = MAX_PINCH - EDGE_DEAD_ZONE

                    # Escala actual (global)
                    if active_model == "CuboUnico":
                        current_scale = cube.scale
                    else:
                        current_scale = composite.scale

                    if pinch_dist <= MID_LOW:
                        # Dedos muy juntos -> escala mínima
                        new_scale = MIN_SCALE
                        last_pinch_dist_mid = None
                        last_zoom_change_time_mid = None
                        zoom_frozen = False
                    elif pinch_dist >= MID_HIGH:
                        # Mano muy abierta -> escala máxima
                        new_scale = MAX_SCALE
                        last_pinch_dist_mid = None
                        last_zoom_change_time_mid = None
                        zoom_frozen = False
                    else:
                        # Zona intermedia -> mapeo + congelado
                        if last_pinch_dist_mid is None:
                            last_pinch_dist_mid = pinch_dist
                            last_zoom_change_time_mid = now
                            zoom_frozen = False
                            new_scale = current_scale
                        else:
                            delta = abs(pinch_dist - last_pinch_dist_mid)

                            if zoom_frozen:
                                # Solo un cambio grande "descongela"
                                if delta > LARGE_MOVEMENT_EPS:
                                    zoom_frozen = False
                                    last_zoom_change_time_mid = now
                                last_pinch_dist_mid = pinch_dist
                                new_scale = current_scale
                            else:
                                if delta > MOVEMENT_EPS:
                                    t = (pinch_dist - MIN_PINCH) / (MAX_PINCH - MIN_PINCH)
                                    t = max(0.0, min(1.0, t))
                                    new_scale = MIN_SCALE + t * (MAX_SCALE - MIN_SCALE)
                                    last_pinch_dist_mid = pinch_dist
                                    last_zoom_change_time_mid = now
                                else:
                                    # Sin movimiento relevante -> posible congelado
                                    if (last_zoom_change_time_mid is not None and
                                            now - last_zoom_change_time_mid > FREEZE_SECS):
                                        zoom_frozen = True
                                    last_pinch_dist_mid = pinch_dist
                                    new_scale = current_scale

                    # Aplicar zoom al modelo correspondiente
                    if active_model == "CuboUnico":
                        cube.scale = new_scale
                    else:
                        # 4 Cubos: zoom global o por pieza seleccionada
                        if (comp_selected_index is not None and
                                0 <= comp_selected_index < len(composite.pieces) and
                                not composite.pieces[comp_selected_index].attached):
                            piece = composite.pieces[comp_selected_index]
                            if not hasattr(piece, "scale_factor"):
                                piece.scale_factor = 1.0
                            base_scale = composite.scale or 1e-3
                            piece.scale_factor = new_scale / base_scale
                        else:
                            composite.scale = new_scale

                    print(
                        f"[LEFT ZOOM] model={active_model} "
                        f"pinch_dist={pinch_dist:.4f} scale={new_scale:.3f} "
                        f"frozen={zoom_frozen}"
                    )
                else:
                    # Sin pinch válido o zoom deshabilitado -> limpiar buffers
                    last_pinch_dist_mid = None
                    last_zoom_change_time_mid = None
                    zoom_frozen = False
        else:
            # No mano izquierda o no hay modelo activo
            last_pinch_dist_mid = None
            last_zoom_change_time_mid = None
            zoom_frozen = False

        # =====================================================
        # 2) MANO DERECHA: menú + interacción + gesto grupo
        # =====================================================
        if right_lm is not None:
            hand_landmarks, lm = right_lm

            index_tip = hand_landmarks.landmark[lm.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[lm.MIDDLE_FINGER_TIP]
            ring_tip = hand_landmarks.landmark[lm.RING_FINGER_TIP]
            pinky_tip = hand_landmarks.landmark[lm.PINKY_TIP]

            index_pip = hand_landmarks.landmark[lm.INDEX_FINGER_PIP]
            middle_pip = hand_landmarks.landmark[lm.MIDDLE_FINGER_PIP]
            ring_pip = hand_landmarks.landmark[lm.RING_FINGER_PIP]
            pinky_pip = hand_landmarks.landmark[lm.PINKY_PIP]

            index_up_r = finger_up(index_tip, index_pip)
            middle_up_r = finger_up(middle_tip, middle_pip)
            ring_up_r = finger_up(ring_tip, ring_pip)
            pinky_up_r = finger_up(pinky_tip, pinky_pip)

            count_up_r = sum([index_up_r, middle_up_r, ring_up_r, pinky_up_r])

            index_only_r = index_up_r and not (middle_up_r or ring_up_r or pinky_up_r)
            open_hand_r = (count_up_r == 4)

            index_norm_r = np.array([index_tip.x, index_tip.y], dtype=np.float32)
            middle_norm_r = np.array([middle_tip.x, middle_tip.y], dtype=np.float32)
            pair_dist = float(np.linalg.norm(index_norm_r - middle_norm_r))
            grip_close = pair_dist < GRIP_DIST_MAX

            ix = int(index_tip.x * w)
            iy = int(index_tip.y * h)
            mx = int(middle_tip.x * w)
            my = int(middle_tip.y * h)
            avg_x = 0.5 * (ix + mx)
            avg_y = 0.5 * (iy + my)

            wrist = hand_landmarks.landmark[lm.WRIST]
            wrist_norm = (wrist.x, wrist.y)

            # ---------- Estado mano para gesto de grupo ----------
            if open_hand_r:
                right_state = "open"
            elif count_up_r == 0:
                right_state = "closed"
            else:
                right_state = "other"

            # Gesto de grupo: open -> closed en 4 Cubos
            if active_model == "Cubo4":
                if (right_state == "closed" and right_state_prev == "open" and
                        now - right_state_change_time < RIGHT_TOGGLE_WINDOW):
                    comp_selected_index = None  # bloque completo
                    zoom_enabled = False
                    last_pinch_dist_mid = None
                    last_zoom_change_time_mid = None
                    zoom_frozen = False
                    print("[4C GROUP] Grupo completo seleccionado (open->closed).")

            if right_state != right_state_prev:
                right_state_prev = right_state
                right_state_change_time = now

            # ---------- Menú / header ----------
            inside_cu_btn = (CU_BTN_rect[0] <= ix <= CU_BTN_rect[2] and
                             CU_BTN_rect[1] <= iy <= CU_BTN_rect[3])
            inside_4c_btn = (FOUR_BTN_rect[0] <= ix <= FOUR_BTN_rect[2] and
                             FOUR_BTN_rect[1] <= iy <= FOUR_BTN_rect[3])
            inside_back_btn = (active_model is not None and
                               BACK_BTN_rect[0] <= ix <= BACK_BTN_rect[2] and
                               BACK_BTN_rect[1] <= iy <= BACK_BTN_rect[3])

            if index_only_r:
                if active_model is None:
                    # Selección de modelo en menú
                    if inside_cu_btn:
                        if menu_hover_start_cubo is None:
                            menu_hover_start_cubo = now
                        elif now - menu_hover_start_cubo >= MENU_HOLD_SECS:
                            active_model = "CuboUnico"
                            cube.offset_2d[:] = 0.0
                            zoom_enabled = False
                            last_pinch_dist_mid = None
                            last_zoom_change_time_mid = None
                            zoom_frozen = False
                            menu_hover_start_cubo = None
                            menu_hover_start_4c = None
                            back_hover_start = None
                            print("[MENU] Modelo 'CuboUnico' activado.")
                    else:
                        menu_hover_start_cubo = None

                    if inside_4c_btn:
                        if menu_hover_start_4c is None:
                            menu_hover_start_4c = now
                        elif now - menu_hover_start_4c >= MENU_HOLD_SECS:
                            active_model = "Cubo4"
                            composite.group_offset_2d[:] = 0.0
                            comp_selected_index = None
                            comp_group_dragging = False
                            comp_piece_dragging = False
                            comp_rotating = False
                            comp_rot_target = None
                            comp_hold_index = None
                            comp_hold_start = None
                            zoom_enabled = False
                            last_pinch_dist_mid = None
                            last_zoom_change_time_mid = None
                            zoom_frozen = False
                            menu_hover_start_cubo = None
                            menu_hover_start_4c = None
                            back_hover_start = None
                            print("[MENU] Modelo '4 Cubos' activado.")
                    else:
                        menu_hover_start_4c = None
                else:
                    # Botón regresar
                    menu_hover_start_cubo = None
                    menu_hover_start_4c = None
                    if inside_back_btn:
                        if back_hover_start is None:
                            back_hover_start = now
                        elif now - back_hover_start >= MENU_HOLD_SECS:
                            print("[MENU] Regresar al menú.")
                            active_model = None
                            back_hover_start = None
                            single_dragging = False
                            single_rotating = False
                            comp_group_dragging = False
                            comp_piece_dragging = False
                            comp_rotating = False
                            comp_selected_index = None
                            zoom_enabled = False
                            last_pinch_dist_mid = None
                            last_zoom_change_time_mid = None
                            zoom_frozen = False
                    else:
                        back_hover_start = None
            else:
                if active_model is None:
                    menu_hover_start_cubo = None
                    menu_hover_start_4c = None
                back_hover_start = None

            # Si no hay modelo, terminamos aquí
            if active_model is None:
                scale_value = 0.0
            else:
                # ================================================
                # CUBO ÚNICO
                # ================================================
                if active_model == "CuboUnico":
                    cv2.circle(frame, (ix, iy), 8, (255, 0, 0), -1)
                    cv2.circle(frame, (mx, my), 8, (255, 0, 0), -1)

                    if interaction_paused:
                        single_dragging = False
                        single_rotating = False
                        single_drag_start_avg = None
                        single_drag_start_offset = None
                        single_rot_start_pos = None
                        single_rot_start_angles = None
                    else:
                        if open_hand_r:
                            # Rotación
                            if single_dragging:
                                single_dragging = False
                                single_drag_start_avg = None
                                single_drag_start_offset = None
                            if not single_rotating:
                                single_rotating = True
                                single_rot_start_pos = wrist_norm
                                single_rot_start_angles = (cube.angle_x, cube.angle_y)
                            else:
                                dxn = wrist_norm[0] - single_rot_start_pos[0]
                                dyn = wrist_norm[1] - single_rot_start_pos[1]
                                cube.angle_y = single_rot_start_angles[1] - dxn * 2.0 * math.pi
                                cube.angle_x = single_rot_start_angles[0] - dyn * 2.0 * math.pi
                        else:
                            if single_rotating:
                                single_rotating = False
                                single_rot_start_pos = None
                                single_rot_start_angles = None

                            # Drag índice+medio
                            if grip_close:
                                if not single_dragging:
                                    inside = False
                                    if getattr(cube, "last_bbox", None) is not None:
                                        mnx, mxx, mny, mxy = cube.last_bbox
                                        if (mnx <= ix <= mxx and mnx <= mx <= mxx and
                                            mny <= iy <= mxy and mny <= my <= mxy):
                                            inside = True
                                    if inside:
                                        single_dragging = True
                                        single_drag_start_avg = (avg_x, avg_y)
                                        single_drag_start_offset = cube.offset_2d.copy()
                                else:
                                    dx = avg_x - single_drag_start_avg[0]
                                    dy = avg_y - single_drag_start_avg[1]
                                    cube.offset_2d = single_drag_start_offset + np.array(
                                        [dx, dy], dtype=np.float32
                                    )
                            else:
                                if single_dragging:
                                    single_dragging = False
                                    single_drag_start_avg = None
                                    single_drag_start_offset = None

                # ================================================
                # 4 CUBOS (bloque + piezas)
                # ================================================
                elif active_model == "Cubo4":
                    cv2.circle(frame, (ix, iy), 8, (255, 0, 0), -1)
                    cv2.circle(frame, (mx, my), 8, (255, 0, 0), -1)

                    # --------- Selección / detach 2s (siempre) ---------
                    if index_only_r and iy > HEADER_H:
                        candidate = None
                        for j, p in enumerate(composite.pieces):
                            if p.last_bbox is None:
                                continue
                            mnx, mxx, mny, mxy = p.last_bbox
                            if mnx <= ix <= mxx and mny <= iy <= mxy:
                                candidate = j
                                break

                        if candidate is not None:
                            if comp_hold_index != candidate:
                                comp_hold_index = candidate
                                comp_hold_start = now
                            elif (comp_hold_start is not None and
                                  now - comp_hold_start >= SELECT_HOLD_SECS):
                                piece = composite.pieces[candidate]
                                if piece.attached:
                                    composite.detach_piece(candidate)
                                comp_selected_index = candidate
                                comp_hold_start = None
                                zoom_enabled = False
                                last_pinch_dist_mid = None
                                last_zoom_change_time_mid = None
                                zoom_frozen = False
                                print(f"[4C] Pieza {candidate} seleccionada.")
                        else:
                            comp_hold_index = None
                            comp_hold_start = None
                    else:
                        comp_hold_index = None
                        comp_hold_start = None

                    # Objetivo actual: bloque o pieza
                    target_block = (comp_selected_index is None)
                    target_piece = (comp_selected_index is not None and
                                    0 <= comp_selected_index < len(composite.pieces))

                    # --------- Drag (bloque / pieza) ---------
                    if not interaction_paused:
                        if target_block:
                            # Drag del bloque
                            if grip_close and not open_hand_r:
                                if not comp_group_dragging:
                                    inside_any = False
                                    for p in composite.pieces:
                                        if p.last_bbox is None:
                                            continue
                                        mnx, mxx, mny, mxy = p.last_bbox
                                        if (mnx <= ix <= mxx and mnx <= mx <= mxx and
                                            mny <= iy <= mxy and mny <= my <= mxy):
                                            inside_any = True
                                            break
                                    if inside_any:
                                        comp_group_dragging = True
                                        comp_group_drag_last_avg = (avg_x, avg_y)
                                else:
                                    dx = avg_x - comp_group_drag_last_avg[0]
                                    dy = avg_y - comp_group_drag_last_avg[1]
                                    composite.move_group(dx, dy)
                                    comp_group_drag_last_avg = (avg_x, avg_y)
                            else:
                                comp_group_dragging = False
                                comp_group_drag_last_avg = None
                        else:
                            comp_group_dragging = False
                            comp_group_drag_last_avg = None

                    # Drag de pieza seleccionada (funciona con o sin pausa)
                    if target_piece:
                        piece = composite.pieces[comp_selected_index]
                        if not piece.attached and grip_close and not open_hand_r:
                            if not comp_piece_dragging:
                                if piece.last_bbox is not None:
                                    mnx, mxx, mny, mxy = piece.last_bbox
                                    if (mnx <= ix <= mxx and mnx <= mx <= mxx and
                                        mny <= iy <= mxy and mny <= my <= mxy):
                                        comp_piece_dragging = True
                                        comp_piece_drag_last_avg = (avg_x, avg_y)
                            else:
                                dx = avg_x - comp_piece_drag_last_avg[0]
                                dy = avg_y - comp_piece_drag_last_avg[1]
                                composite.move_piece(comp_selected_index, dx, dy)
                                comp_piece_drag_last_avg = (avg_x, avg_y)
                        else:
                            comp_piece_dragging = False
                            comp_piece_drag_last_avg = None
                    else:
                        comp_piece_dragging = False
                        comp_piece_drag_last_avg = None

                    # --------- Rotación (bloque / pieza) ---------
                    if open_hand_r:
                        if interaction_paused and target_block:
                            # en pausa y sin pieza seleccionada -> sin rotación
                            pass
                        else:
                            if not comp_rotating:
                                comp_rotating = True
                                comp_rot_start_pos = wrist_norm
                                if target_block or (target_piece and composite.pieces[comp_selected_index].attached):
                                    comp_rot_target = "block"
                                    comp_rot_block_start = (composite.angle_x,
                                                            composite.angle_y)
                                else:
                                    comp_rot_target = "piece"
                                    comp_rot_piece_index = comp_selected_index
                                    piece = composite.pieces[comp_selected_index]
                                    comp_rot_piece_start = piece.local_angles.copy()
                            else:
                                dxn = wrist_norm[0] - comp_rot_start_pos[0]
                                dyn = wrist_norm[1] - comp_rot_start_pos[1]
                                if comp_rot_target == "block":
                                    composite.angle_y = comp_rot_block_start[1] - dxn * 2.0 * math.pi
                                    composite.angle_x = comp_rot_block_start[0] - dyn * 2.0 * math.pi
                                elif comp_rot_target == "piece" and comp_rot_piece_index is not None:
                                    j = comp_rot_piece_index
                                    if 0 <= j < len(composite.pieces):
                                        piece = composite.pieces[j]
                                        piece.local_angles[1] = (
                                            comp_rot_piece_start[1] - dxn * 2.0 * math.pi
                                        )
                                        piece.local_angles[0] = (
                                            comp_rot_piece_start[0] - dyn * 2.0 * math.pi
                                        )
                    else:
                        if comp_rotating:
                            comp_rotating = False
                            comp_rot_target = None
                            comp_rot_piece_index = None
                            comp_rot_start_pos = None
                            comp_rot_block_start = None
                            comp_rot_piece_start = None

        else:
            # Sin mano derecha: apagar interacciones activas
            single_dragging = False
            single_drag_start_avg = None
            single_drag_start_offset = None
            single_rotating = False
            single_rot_start_pos = None
            single_rot_start_angles = None

            comp_group_dragging = False
            comp_group_drag_last_avg = None
            comp_piece_dragging = False
            comp_piece_drag_last_avg = None
            comp_rotating = False
            comp_rot_target = None
            comp_rot_piece_index = None
            comp_rot_start_pos = None
            comp_rot_block_start = None
            comp_rot_piece_start = None
            comp_hold_index = None
            comp_hold_start = None

        # =====================================================
        # 3) Dibujar modelos
        # =====================================================
        if active_model == "CuboUnico":
            cube.draw_cube(frame)
            scale_value = cube.scale
        elif active_model == "Cubo4":
            composite.draw(frame)
            scale_value = composite.scale
            # Resaltar pieza seleccionada
            if comp_selected_index is not None:
                p = composite.pieces[comp_selected_index]
                if p.last_bbox is not None:
                    mnx, mxx, mny, mxy = p.last_bbox
                    cv2.rectangle(
                        frame,
                        (int(mnx), int(mny)),
                        (int(mxx), int(mxy)),
                        (0, 255, 255),
                        3
                    )
        else:
            scale_value = 0.0

        # =====================================================
        # 4) Header y botones
        # =====================================================
        cv2.rectangle(frame, (0, HEADER_Y0), (w, HEADER_Y1), (40, 40, 40), -1)

        if active_model is None:
            # CuboUnico
            cv2.rectangle(frame, (CU_BTN_X, CU_BTN_Y),
                          (CU_BTN_X + CU_BTN_W, CU_BTN_Y + CU_BTN_H),
                          (70, 70, 200), -1)
            cv2.rectangle(frame, (CU_BTN_X, CU_BTN_Y),
                          (CU_BTN_X + CU_BTN_W, CU_BTN_Y + CU_BTN_H),
                          (255, 255, 255), 2)
            if menu_hover_start_cubo is not None:
                elapsed = now - menu_hover_start_cubo
                progress = max(0.0, min(1.0, elapsed / MENU_HOLD_SECS))
                fill_w = int(CU_BTN_W * progress)
                cv2.rectangle(frame, (CU_BTN_X, CU_BTN_Y),
                              (CU_BTN_X + fill_w, CU_BTN_Y + CU_BTN_H),
                              (100, 255, 100), -1)
            cv2.putText(frame, "CuboUnico",
                        (CU_BTN_X + 10, CU_BTN_Y + int(CU_BTN_H * 0.7)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)

            # 4 Cubos
            cv2.rectangle(frame, (FOUR_BTN_X, FOUR_BTN_Y),
                          (FOUR_BTN_X + FOUR_BTN_W, FOUR_BTN_Y + FOUR_BTN_H),
                          (70, 200, 70), -1)
            cv2.rectangle(frame, (FOUR_BTN_X, FOUR_BTN_Y),
                          (FOUR_BTN_X + FOUR_BTN_W, FOUR_BTN_Y + FOUR_BTN_H),
                          (255, 255, 255), 2)
            if menu_hover_start_4c is not None:
                elapsed = now - menu_hover_start_4c
                progress = max(0.0, min(1.0, elapsed / MENU_HOLD_SECS))
                fill_w = int(FOUR_BTN_W * progress)
                cv2.rectangle(frame, (FOUR_BTN_X, FOUR_BTN_Y),
                              (FOUR_BTN_X + fill_w, FOUR_BTN_Y + FOUR_BTN_H),
                              (100, 255, 100), -1)
            cv2.putText(frame, "4 Cubos",
                        (FOUR_BTN_X + 10, FOUR_BTN_Y + int(FOUR_BTN_H * 0.7)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)
        else:
            # Botón regresar
            cv2.rectangle(frame, (BACK_BTN_X, BACK_BTN_Y),
                          (BACK_BTN_X + BACK_BTN_W, BACK_BTN_Y + BACK_BTN_H),
                          (200, 70, 70), -1)
            cv2.rectangle(frame, (BACK_BTN_X, BACK_BTN_Y),
                          (BACK_BTN_X + BACK_BTN_W, BACK_BTN_Y + BACK_BTN_H),
                          (255, 255, 255), 2)
            if back_hover_start is not None:
                elapsed = now - back_hover_start
                progress = max(0.0, min(1.0, elapsed / MENU_HOLD_SECS))
                fill_w = int(BACK_BTN_W * progress)
                cv2.rectangle(frame, (BACK_BTN_X, BACK_BTN_Y),
                              (BACK_BTN_X + fill_w, BACK_BTN_Y + BACK_BTN_H),
                              (255, 200, 100), -1)
            cv2.putText(frame, "Regresar",
                        (BACK_BTN_X + 10, BACK_BTN_Y + int(BACK_BTN_H * 0.7)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)

        # Texto de escala
        cv2.putText(frame, f"Scale: {scale_value:.2f}",
                    (10, HEADER_H + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2)

        if interaction_paused and active_model is not None:
            cv2.putText(frame,
                        "PAUSA: indice izq solo (pulgar lejos/no visible) sobre modelo",
                        (10, HEADER_H + 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 2)

        cv2.imshow("3D-4U - Menu + Modelos 3D", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

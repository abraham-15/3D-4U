# -*- coding: utf-8 -*-
"""
Control gestual para 3D-4U (dos manos)
--------------------------------------
- Mano izquierda:
    * Zoom in/out usando la distancia entre pulgar (THUMB_TIP) e índice (INDEX_FINGER_TIP).
    * Se dibuja una línea verde entre pulgar e índice cuando están dentro del umbral.
    * Cuando la separación supera el umbral, la línea desaparece y se resetea el zoom.
- Mano derecha:
    * Interacción usando índice + medio.
    * Cuando índice y medio están separados (gesto en "V"), el movimiento del índice
      genera eventos de interacción (rotación/traslación).
    * Visualmente SOLO se ven dos puntos azules en índice y medio (sin líneas ni texto).
"""

import cv2
import mediapipe as mp
import numpy as np


class HandGestureController:
    def __init__(
        self,
        pinch_threshold=0.05,          # Umbral para pellizco (izquierda)
        zoom_move_epsilon=0.001,       # Cambio mínimo en distancia para zoom
        move_threshold=0.01,           # Umbral mínimo de movimiento del índice (derecha)
        interact_spread_threshold=0.04 # Distancia mínima índice-medio para "V" (derecha)
    ):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,           # Hasta 2 manos
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.drawer = mp.solutions.drawing_utils

        # Parámetros
        self.pinch_threshold = pinch_threshold
        self.zoom_move_epsilon = zoom_move_epsilon
        self.move_threshold = move_threshold
        self.interact_spread_threshold = interact_spread_threshold

        # Estado mano izquierda (zoom)
        self.prev_left_pinch_dist = None

        # Estado mano derecha (interacción)
        self.prev_right_index_pos = None   # (x_norm, y_norm)
        self.right_interact_active = False

    # ==========================
    #  Funciones de ayuda
    # ==========================
    def _get_landmark_xy(self, landmark, img_shape):
        """Convierte un landmark normalizado a coordenadas de píxel (x, y)."""
        h, w, _ = img_shape
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        return x, y

    def _euclidean_distance_norm(self, p1, p2):
        """Distancia euclidiana normalizada (0-1) usando coordenadas normalizadas."""
        return float(np.linalg.norm(np.array(p1) - np.array(p2)))

    # ==========================
    #  Callbacks / “eventos”
    # ==========================
    def on_zoom(self, zoom_delta):
        """
        Evento de zoom (mano izquierda).
        zoom_delta > 0  -> acercar (zoom in)
        zoom_delta < 0  -> alejar (zoom out)
        """
        if zoom_delta > 0:
            print(f"[ZOOM IN] delta: {zoom_delta:.4f}")
        elif zoom_delta < 0:
            print(f"[ZOOM OUT] delta: {zoom_delta:.4f}")

    def on_rotate(self, dx, dy):
        """
        Evento de rotación/traslación basado en movimiento del índice (mano derecha).
        dx, dy son desplazamientos normalizados.
        """
        print(f"[RIGHT INTERACT] dx={dx:.4f}, dy={dy:.4f}")

    # ==========================
    #  Lógica mano izquierda (zoom)
    # ==========================
    def _process_left_hand(self, frame_bgr, hand_landmarks):
        """
        Mano izquierda -> controla ZOOM:
          - Mide distancia pulgar-índice.
          - Mientras la distancia sea menor a pinch_threshold:
              * Dibuja línea verde.
              * Usa el cambio en esa distancia para generar zoom_delta.
          - Cuando la distancia supera pinch_threshold:
              * No dibuja línea.
              * Resetea referencia de distancia.
        """
        lm = self.mp_hands.HandLandmark
        thumb_tip = hand_landmarks.landmark[lm.THUMB_TIP]          # 4
        index_tip = hand_landmarks.landmark[lm.INDEX_FINGER_TIP]   # 8

        thumb_norm = (thumb_tip.x, thumb_tip.y)
        index_norm = (index_tip.x, index_tip.y)

        # Distancia normalizada entre pulgar e índice
        pinch_dist = self._euclidean_distance_norm(thumb_norm, index_norm)

        # Coordenadas para dibujar (píxeles)
        tx, ty = self._get_landmark_xy(thumb_tip, frame_bgr.shape)
        ix, iy = self._get_landmark_xy(index_tip, frame_bgr.shape)

        # Solo consideramos pellizco si la distancia es pequeña
        if pinch_dist < self.pinch_threshold:
            # --- DEBUG EN TERMINAL: mano izquierda ---
            print(
                f"[LEFT] thumb_px=({tx:4d},{ty:4d}) "
                f"index_px=({ix:4d},{iy:4d}) "
                f"dist_norm={pinch_dist:.4f}"
            )

            # Dibujar línea verde (zoom visual)
            cv2.line(frame_bgr, (tx, ty), (ix, iy), (0, 255, 0), 2)
            cv2.putText(
                frame_bgr,
                f"LEFT ZOOM dist: {pinch_dist:.3f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            # Calcular delta para zoom
            if self.prev_left_pinch_dist is not None:
                zoom_delta = self.prev_left_pinch_dist - pinch_dist
                if abs(zoom_delta) > self.zoom_move_epsilon:
                    self.on_zoom(zoom_delta)

            # Actualizar distancia previa
            self.prev_left_pinch_dist = pinch_dist
        else:
            # Sin pellizco: no hay línea y se resetea referencia
            self.prev_left_pinch_dist = None

    # ==========================
    #  Lógica mano derecha (interacción)
    # ==========================
    def _process_right_hand(self, frame_bgr, hand_landmarks):
        """
        Mano derecha -> interacción (rotación/traslación).
        - Usa índice + medio.
        - Cuando spread_dist > interact_spread_threshold (gesto en "V"):
            * Activa interacción.
            * El movimiento del índice genera eventos on_rotate.
        - Visualmente SOLO:
            * Dos puntos azules en índice y medio (sin líneas ni texto).
        """
        lm = self.mp_hands.HandLandmark
        index_tip = hand_landmarks.landmark[lm.INDEX_FINGER_TIP]   # 8
        middle_tip = hand_landmarks.landmark[lm.MIDDLE_FINGER_TIP] # 12

        index_norm = (index_tip.x, index_tip.y)
        middle_norm = (middle_tip.x, middle_tip.y)

        # Distancia entre índice y medio
        spread_dist = self._euclidean_distance_norm(index_norm, middle_norm)

        # Coordenadas para dibujar puntos (píxeles)
        ix, iy = self._get_landmark_xy(index_tip, frame_bgr.shape)
        mx, my = self._get_landmark_xy(middle_tip, frame_bgr.shape)

        # Puntos azules en índice y medio (siempre visibles)
        cv2.circle(frame_bgr, (ix, iy), 8, (255, 0, 0), -1)
        cv2.circle(frame_bgr, (mx, my), 8, (255, 0, 0), -1)

        # ¿Gestos en "V"? (índice y medio separados)
        if spread_dist > self.interact_spread_threshold:
            # Activar interacción
            if not self.right_interact_active:
                print("[RIGHT] Interacción ACTIVADA (V con índice+medio)")
                self.right_interact_active = True
                self.prev_right_index_pos = index_norm

            # --- DEBUG EN TERMINAL: mano derecha ---
            print(
                f"[RIGHT] index_px=({ix:4d},{iy:4d}) "
                f"middle_px=({mx:4d},{my:4d})"
            )

            # Movimiento del índice -> interacción
            if self.prev_right_index_pos is not None:
                dx = index_norm[0] - self.prev_right_index_pos[0]
                dy = index_norm[1] - self.prev_right_index_pos[1]
                dist_move = np.sqrt(dx**2 + dy**2)

                if dist_move > self.move_threshold:
                    self.on_rotate(dx, dy)

            # Actualizar posición previa
            self.prev_right_index_pos = index_norm
        else:
            # Dedos juntos o casi juntos -> desactivar interacción
            if self.right_interact_active:
                print("[RIGHT] Interacción DESACTIVADA (V cerrada)")
            self.right_interact_active = False
            self.prev_right_index_pos = None

    # ==========================
    #  Procesamiento por frame
    # ==========================
    def process_frame(self, frame_bgr):
        """
        Procesa un frame BGR de OpenCV.
        Devuelve el frame con anotaciones.
        """
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self.hands.process(img_rgb)

        left_seen = False
        right_seen = False

        if result.multi_hand_landmarks and result.multi_handedness:
            for hand_landmarks, handedness in zip(
                result.multi_hand_landmarks, result.multi_handedness
            ):
                # Dibujar esqueleto de la mano (huesitos)
                self.drawer.draw_landmarks(
                    frame_bgr,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )

                # Etiqueta: 'Left' o 'Right'
                label = handedness.classification[0].label  # 'Left' o 'Right'

                if label == 'Left':
                    left_seen = True
                    self._process_left_hand(frame_bgr, hand_landmarks)
                elif label == 'Right':
                    right_seen = True
                    self._process_right_hand(frame_bgr, hand_landmarks)

        # Si en este frame no se vio la mano izquierda, reseteamos solo su estado
        if not left_seen:
            self.prev_left_pinch_dist = None

        # Si no se vio la derecha, reseteamos solo su estado
        if not right_seen:
            if self.right_interact_active:
                print("[RIGHT] Interacción DESACTIVADA (mano no detectada)")
            self.right_interact_active = False
            self.prev_right_index_pos = None

        return frame_bgr


def main():
    cap = cv2.VideoCapture(0)  # 0 = webcam por defecto

    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return

    controller = HandGestureController(
        pinch_threshold=0.40,          # Ajusta si tu pellizco no se detecta bien
        zoom_move_epsilon=0.001,       # Sensibilidad del zoom
        move_threshold=0.01,           # Sensibilidad de movimiento del índice derecho
        interact_spread_threshold=0.04 # Qué tan abierta debe estar la "V" índice+medio
    )

    print("Presiona 'q' para salir.")
    print("- Mano IZQUIERDA: pellizco pulgar-índice -> zoom (línea verde mientras están cerca).")
    print("- Mano DERECHA : índice+medio en 'V' -> interacción (solo puntos azules).")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # espejo horizontal para que se vea natural
        frame_processed = controller.process_frame(frame)

        cv2.imshow("3D-4U - Control dos manos", frame_processed)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

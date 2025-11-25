# cube_renderer.py
# Clase que renderiza un cubo 3D morado con rotación, escala y offset 2D.

import cv2
import numpy as np
import math


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

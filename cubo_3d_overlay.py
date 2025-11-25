# -*- coding: utf-8 -*-
"""
Cubo 3D morado sobre la cámara para 3D-4U
-----------------------------------------
- Muestra el video de la webcam.
- Superpone un cubo 3D con cada cara en un tono distinto de morado.
- El cubo rota suavemente para comprobar la proyección 3D->2D.
"""

import cv2
import numpy as np
import math


class PurpleCubeRenderer:
    def __init__(self, size=1.0, distance=5.0):
        """
        Args:
            size: tamaño básico del cubo (en unidades arbitrarias).
            distance: distancia de la cámara al centro del cubo (para proyección).
        """
        self.size = size
        self.distance = distance

        # Vertices de un cubo centrado en el origen
        # [-1,1] escalado por size
        s = self.size
        self.vertices = np.array([
            [-s, -s, -s],
            [-s, -s,  s],
            [-s,  s, -s],
            [-s,  s,  s],
            [ s, -s, -s],
            [ s, -s,  s],
            [ s,  s, -s],
            [ s,  s,  s],
        ], dtype=np.float32)

        # Caras como listas de índices de vertices (quads)
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
            (224, 96, 224),  # casi rosa/magenta
        ]

        # Ángulos de rotación iniciales
        self.angle_x = 0.0
        self.angle_y = 0.0
        self.angle_z = 0.0

    # ==========================
    #  Transformaciones 3D
    # ==========================
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
        Proyección de puntos 3D al plano de la imagen con perspectiva sencilla.

        Args:
            points_3d: array (N, 3)
            frame_shape: shape del frame (h, w, c)
            f: distancia focal (puedes ajustarla)

        Returns:
            points_2d: array (N, 2) en píxeles (float32)
            depths: array (N,) con z (para ordenamiento de caras)
        """
        h, w, _ = frame_shape
        cx, cy = w / 2.0, h / 2.0

        # Trasladar el cubo "al frente" de la cámara
        translated = points_3d.copy()
        translated[:, 2] += self.distance  # mover en z

        # Evitar divisiones por cero
        z = translated[:, 2].copy()
        z[z == 0] = 1e-6

        x = translated[:, 0]
        y = translated[:, 1]

        # Proyección perspectiva
        u = f * (x / z) + cx
        v = f * (-y / z) + cy  # signo - para invertir eje Y

        points_2d = np.stack([u, v], axis=1).astype(np.float32)
        depths = z
        return points_2d, depths

    # ==========================
    #  Dibujo
    # ==========================
    def draw_cube(self, frame):
        """Dibuja el cubo rotado sobre el frame dado."""

        # 1) Rotar vertices
        R = self._rotation_matrix(self.angle_x, self.angle_y, self.angle_z)
        verts_rot = self.vertices @ R.T

        # 2) Proyectar a 2D
        pts_2d, depths = self._project_points(verts_rot, frame.shape, f=800.0)

        # 3) Ordenar caras por profundidad (pintar de atrás hacia adelante)
        face_depths = []
        for i, face in enumerate(self.faces):
            z_mean = depths[face].mean()
            face_depths.append((z_mean, i))
        face_depths.sort(reverse=True)  # más lejos primero

        # 4) Dibujar caras sólidas
        for _, face_idx in face_depths:
            face = self.faces[face_idx]
            color = self.face_colors[face_idx % len(self.face_colors)]

            pts_face = pts_2d[face].astype(np.int32)
            # cv2.fillConvexPoly requiere shape (N,1,2)
            pts_face = pts_face.reshape((-1, 1, 2))

            cv2.fillConvexPoly(frame, pts_face, color)

            # Opcional: dibujar borde
            cv2.polylines(frame, [pts_face], isClosed=True, color=(255, 255, 255), thickness=1)

        # 5) Actualizar ángulos para animación
        self.angle_x += 0.01
        self.angle_y += 0.02
        # si no quieres giro en Z, deja angle_z en 0
        # self.angle_z += 0.015


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return

    cube = PurpleCubeRenderer(size=1.5, distance=6.0)

    print("Mostrando cámara con cubo 3D superpuesto. Presiona 'q' para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # espejo para que se vea natural

        # Dibujar cubo 3D sobre el frame
        cube.draw_cube(frame)

        cv2.imshow("3D-4U - Cubo 3D sobre cámara", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

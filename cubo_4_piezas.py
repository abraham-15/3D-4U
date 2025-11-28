# -*- coding: utf-8 -*-
"""
cubo_4_piezas.py

Bloque 3D 2x2x2 (8 cubos individuales)
--------------------------------------
- 8 cubos pequeños forman un único "CUBO GENERAL" 2x2x2.
- Cada cubo es un objeto independiente, pero en inicio se comportan como
  un solo bloque:
    * mismo zoom (CompositeCubes.scale)
    * mismo movimiento (CompositeCubes.group_offset_2d)
    * misma rotación global (CompositeCubes.angle_x/angle_y/angle_z)

- Se puede SEPARAR un cubo (attached=False).
- Los cubos separados se mueven en 2D con move_piece evitando "encimarse".
"""

import cv2
import numpy as np
import math


# =====================================================
#  Render básico de cubo 3D
# =====================================================
class SimpleCube3D:
    """
    Renderizador básico de un cubo 3D sólido de un solo color.
    """

    def __init__(self, color_bgr, base_size=0.5, distance=6.0):
        # base_size = 0.5 => cubo pequeño de lado 1 en coordenadas locales
        self.base_size = base_size
        self.distance = distance
        self.color = color_bgr

        s = self.base_size
        # Vértices de un cubo centrado en el origen (lado = 2*s)
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

        # Caras (índices a los vértices)
        self.faces = [
            [0, 1, 3, 2],  # izquierda
            [4, 5, 7, 6],  # derecha
            [0, 1, 5, 4],  # abajo
            [2, 3, 7, 6],  # arriba
            [0, 2, 6, 4],  # fondo
            [1, 3, 7, 5],  # frente
        ]

    @staticmethod
    def rotation_matrix(ax, ay, az):
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

    def draw(self, frame, scale, angle_x, angle_y, angle_z,
             offset_2d, center_unit_3d):
        """
        Dibuja el cubo sobre el frame.

        Parámetros:
        - scale: escala global (o efectiva) de la pieza.
        - angle_x, angle_y, angle_z: rotación.
        - offset_2d: traslación en la imagen.
        - center_unit_3d: centro del cubo en coord. de bloque (ej. ±0.5).

        Devuelve:
        - pts_2d: puntos 2D proyectados (8 x 2).
        - bbox:  (min_x, max_x, min_y, max_y) en píxeles.
        """
        # 1) Sumamos centro 3D en coordenadas de bloque
        verts = self.base_vertices + np.array(center_unit_3d, dtype=np.float32)

        # 2) Escalamos
        verts *= scale

        # 3) Rotamos
        R = self.rotation_matrix(angle_x, angle_y, angle_z)
        verts_rot = verts @ R.T

        # 4) Proyectamos
        pts_2d, depths = self._project_points(verts_rot, frame.shape, f=800.0)

        # 5) Offset 2D
        if offset_2d is not None:
            pts_2d = pts_2d + offset_2d

        # 6) Bounding box
        min_x = float(np.min(pts_2d[:, 0]))
        max_x = float(np.max(pts_2d[:, 0]))
        min_y = float(np.min(pts_2d[:, 1]))
        max_y = float(np.max(pts_2d[:, 1]))
        bbox = (min_x, max_x, min_y, max_y)

        # 7) Ordenar caras por profundidad
        face_depths = []
        for i, face in enumerate(self.faces):
            z_mean = depths[face].mean()
            face_depths.append((z_mean, i))
        face_depths.sort(reverse=True)

        # 8) Dibujar caras
        for _, face_idx in face_depths:
            face = self.faces[face_idx]
            color = self.color
            pts_face = pts_2d[face].astype(np.int32).reshape((-1, 1, 2))
            cv2.fillConvexPoly(frame, pts_face, color)
            cv2.polylines(frame, [pts_face], isClosed=True,
                          color=(255, 255, 255), thickness=1)

        return pts_2d, bbox


# =====================================================
#  Pieza individual del bloque 2x2x2
# =====================================================
class CubePiece:
    """
    Una pieza (cubo pequeño) del bloque 2x2x2.

    - center_unit_3d: centro en coordenadas de bloque (±0.5, ±0.5, ±0.5).
      El bloque completo ocupa el rango aprox. [-1, 1] en cada eje antes de escalar.
    """
    def __init__(self, renderer, center_unit_3d):
        self.renderer = renderer
        self.center_unit_3d = np.array(center_unit_3d, dtype=np.float32)

        # Estado de adjunción al bloque
        self.attached = True
        # Offset 2D cuando está separada
        self.free_offset_2d = np.array([0.0, 0.0], dtype=np.float32)

        # Info 2D (actualizada en draw)
        self.last_pts_2d = None
        self.last_bbox = None

        # Rotación local (para giros individuales)
        self.local_angles = np.zeros(3, dtype=np.float32)  # [ax, ay, az]

        # Escala relativa de la pieza respecto al bloque (para zoom por pieza)
        # 1.0 => mismo tamaño que la escala global del bloque.
        self.scale_factor = 1.0

    def current_offset_2d(self, group_offset_2d):
        """
        Offset 2D actual:
        - adjunto  -> offset del grupo (bloque completo).
        - separado -> free_offset_2d.
        """
        if self.attached:
            return group_offset_2d
        else:
            return self.free_offset_2d

    def draw(self, frame, base_scale, base_angle_x, base_angle_y, base_angle_z,
             group_offset_2d):
        """
        Dibuja la pieza como cubo 3D.
        - base_scale: escala global del bloque (CompositeCubes.scale).
        - scale_factor: escala relativa de la pieza (self.scale_factor).
        """
        offset_2d = self.current_offset_2d(group_offset_2d)

        # 🔹 Escala efectiva = escala global * factor local
        effective_scale = base_scale * getattr(self, "scale_factor", 1.0)

        # Rotación total = global + local
        angle_x = base_angle_x + self.local_angles[0]
        angle_y = base_angle_y + self.local_angles[1]
        angle_z = base_angle_z + self.local_angles[2]

        pts_2d, bbox = self.renderer.draw(
            frame,
            scale=effective_scale,
            angle_x=angle_x,
            angle_y=angle_y,
            angle_z=angle_z,
            offset_2d=offset_2d,
            center_unit_3d=self.center_unit_3d
        )

        self.last_pts_2d = pts_2d
        self.last_bbox = bbox


# =====================================================
#  Bloque 2x2x2 de 8 cubos
# =====================================================
class CompositeCubes:
    """
    Bloque principal formado por 8 cubos (2x2x2).

    - Todas las piezas comparten escala y rotación global.
    - Cada pieza tiene center_unit_3d en {±0.5}^3.
    """

    def __init__(self, initial_scale=0.3):
        self.scale = initial_scale
        self.angle_x = 0.0
        self.angle_y = 0.0
        self.angle_z = 0.0

        # Offset 2D del bloque completo (para moverlo en pantalla)
        self.group_offset_2d = np.array([0.0, 0.0], dtype=np.float32)

        # Posiciones unitarias de los 8 cubos: (±0.5, ±0.5, ±0.5)
        centers_unit3d = []
        for sx in (-0.5, 0.5):
            for sy in (-0.5, 0.5):
                for sz in (-0.5, 0.5):
                    centers_unit3d.append((sx, sy, sz))

        # Colores diferentes
        colors = [
            (0, 0, 255),      # rojo
            (0, 255, 0),      # verde
            (255, 0, 0),      # azul
            (0, 255, 255),    # amarillo
            (255, 0, 255),    # magenta
            (255, 255, 0),    # cian
            (128, 0, 255),    # violeta
            (0, 128, 255),    # naranja-azulado
        ]

        self.pieces = []
        for i, center_u in enumerate(centers_unit3d):
            renderer = SimpleCube3D(
                color_bgr=colors[i % len(colors)],
                base_size=0.5,
                distance=6.0
            )
            piece = CubePiece(renderer, center_unit_3d=center_u)
            self.pieces.append(piece)

    # ---------- Dibujo ----------
    def draw(self, frame):
        """
        Dibuja todo el bloque 2x2x2 sobre 'frame'.
        Se ordenan los cubos por profundidad para que el pintado sea correcto.
        """
        # Matriz de rotación global (solo para calcular profundidad)
        R = SimpleCube3D.rotation_matrix(self.angle_x, self.angle_y, self.angle_z)

        # Calculamos la profundidad (z) del centro de cada cubo
        depth_list = []
        for idx, piece in enumerate(self.pieces):
            center_local = piece.center_unit_3d * self.scale  # tras el zoom global
            center_rot = center_local @ R.T
            z = center_rot[2]
            depth_list.append((z, idx))

        # Ordenamos de más lejano a más cercano (z grande primero)
        depth_list.sort(reverse=True)

        # Dibujamos en ese orden
        for _, idx in depth_list:
            piece = self.pieces[idx]
            piece.draw(
                frame,
                base_scale=self.scale,
                base_angle_x=self.angle_x,
                base_angle_y=self.angle_y,
                base_angle_z=self.angle_z,
                group_offset_2d=self.group_offset_2d
            )

    # ---------- Movimiento conjunto ----------
    def move_group(self, dx, dy):
        """
        Mueve TODO el bloque en 2D (solo afecta a piezas adjuntas).
        """
        delta = np.array([dx, dy], dtype=np.float32)
        self.group_offset_2d += delta

    # ---------- Rotación global ----------
    def rotate_global(self, d_ax, d_ay, d_az=0.0):
        """
        Rotación global del bloque (afecta a los 8 cubos).
        """
        self.angle_x += d_ax
        self.angle_y += d_ay
        self.angle_z += d_az

    # ---------- Ayuda colisiones 2D ----------
    @staticmethod
    def _boxes_intersect(b1, b2):
        """
        Función de ayuda para colisiones 2D.
        b = (min_x, max_x, min_y, max_y)
        """
        x1_min, x1_max, y1_min, y1_max = b1
        x2_min, x2_max, y2_min, y2_max = b2

        if x1_max < x2_min or x2_max < x1_min:
            return False
        if y1_max < y2_min or y2_max < y1_min:
            return False
        return True

    # ---------- Separar una pieza ----------
    def detach_piece(self, index, detach_factor=1.2):
        """
        Separa la pieza 'index' del bloque, moviéndola hacia afuera
        sin encimarse con las demás (en 2D), pero SOLO un poco,
        para que siga visible y cerca del bloque.
        """
        if not (0 <= index < len(self.pieces)):
            return

        piece = self.pieces[index]
        if not piece.attached:
            # Ya estaba separada
            return
        if piece.last_bbox is None:
            # Aún no se ha dibujado
            return

        min_x, max_x, min_y, max_y = piece.last_bbox
        cx = 0.5 * (min_x + max_x)
        cy = 0.5 * (min_y + max_y)

        # Centro aproximado del bloque (promedio de todos los cubos)
        sum_x = 0.0
        sum_y = 0.0
        count = 0
        for p in self.pieces:
            if p.last_bbox is None:
                continue
            mnx, mxx, mny, mxy = p.last_bbox
            pcx = 0.5 * (mnx + mxx)
            pcy = 0.5 * (mny + mxy)
            sum_x += pcx
            sum_y += pcy
            count += 1

        if count > 0:
            bx = sum_x / count
            by = sum_y / count
        else:
            bx, by = cx, cy  # fallback

        # Dirección desde el centro del bloque hacia el cubo
        dir_x = cx - bx
        dir_y = cy - by
        norm = math.hypot(dir_x, dir_y)
        if norm < 1e-3:
            # Si está exactamente en el centro, sácalo hacia arriba
            dir_x, dir_y = 0.0, -1.0
            norm = 1.0
        dir_x /= norm
        dir_y /= norm

        # Tamaño del cubo en píxeles
        width = max_x - min_x
        height = max_y - min_y
        base = max(width, height) * detach_factor

        # Límite máximo para no alejarlo demasiado
        max_extra = max(width, height) * 2.0
        base = min(base, max_extra)

        dx = dir_x * base
        dy = dir_y * base
        cand_bbox = (min_x + dx, max_x + dx, min_y + dy, max_y + dy)

        # Ajuste fino por si aún chocara con algún otro cubo
        for _ in range(3):
            collision = False
            for j, other in enumerate(self.pieces):
                if j == index:
                    continue
                if other.last_bbox is None:
                    continue
                if self._boxes_intersect(cand_bbox, other.last_bbox):
                    collision = True
                    break
            if not collision:
                break
            # Si sigue chocando, incrementamos muy poco la distancia
            base *= 1.1
            base = min(base, max_extra)
            dx = dir_x * base
            dy = dir_y * base
            cand_bbox = (min_x + dx, max_x + dx, min_y + dy, max_y + dy)

        # Marcamos la pieza como separada pero cerca del bloque
        piece.attached = False
        piece.free_offset_2d = self.group_offset_2d + np.array([dx, dy], dtype=np.float32)
        piece.last_bbox = cand_bbox
        print(f"[DETACH] Pieza {index} separada del bloque (dist={base:.1f}px).")

    # ---------- Adjuntar/separar piezas (para futuro) ----------
    def toggle_attach(self, index):
        """
        Alterna entre adjunto / separado.
        Por ahora solo separa; re-enganchar podría implementarse después.
        """
        if not (0 <= index < len(self.pieces)):
            return
        piece = self.pieces[index]
        if piece.attached:
            self.detach_piece(index)
        else:
            piece.attached = True
            piece.free_offset_2d = self.group_offset_2d.copy()
            piece.scale_factor = 1.0
            piece.local_angles[:] = 0.0
            print(f"[ATTACH] Pieza {index} re-encajada en el bloque.")

    # ---------- Mover pieza separada evitando encimarse ----------
    def move_piece(self, index, dx, dy):
        """
        Mueve una pieza SEPARADA en 2D, evitando encimarse con otras.
        """
        if not (0 <= index < len(self.pieces)):
            return

        piece = self.pieces[index]
        if piece.attached or piece.last_bbox is None:
            # Si está adjunta, no se mueve individualmente.
            return

        new_offset = piece.free_offset_2d + np.array([dx, dy], dtype=np.float32)

        # Bounding box tentativa: desplazamos la actual por (dx, dy)
        min_x, max_x, min_y, max_y = piece.last_bbox
        cand_bbox = (min_x + dx, max_x + dx, min_y + dy, max_y + dy)

        # Comprobar colisión con todas las demás piezas
        for j, other in enumerate(self.pieces):
            if j == index:
                continue
            if other.last_bbox is None:
                continue
            if self._boxes_intersect(cand_bbox, other.last_bbox):
                # Colisión: no se aplica el movimiento
                return

        # Sin colisión: aplicar movimiento
        piece.free_offset_2d = new_offset
        piece.last_bbox = cand_bbox


# =====================================================
#  Demo por teclado (opcional)
# =====================================================
def main():
    """
    Pequeña demo para probar el bloque 2x2x2 con teclado.
    (Para tu integración con manos, no es necesario ejecutar esto.)
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return

    composite = CompositeCubes(initial_scale=0.3)

    print("Demo cubo 2x2x2 (8 cubos como un solo objeto + separaciones):")
    print(" W/A/S/D: mover bloque completo")
    print(" Q: salir")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        composite.draw(frame)

        cv2.putText(
            frame,
            "Bloque 2x2x2",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.imshow("Cubo 2x2x2 (demo teclado)", frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), ord('Q')):
            break

        step_group = 10
        if key in (ord('w'), ord('W')):
            composite.move_group(0, -step_group)
        if key in (ord('s'), ord('S')):
            composite.move_group(0, step_group)
        if key in (ord('a'), ord('A')):
            composite.move_group(-step_group, 0)
        if key in (ord('d'), ord('D')):
            composite.move_group(step_group, 0)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

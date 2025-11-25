# -*- coding: utf-8 -*-
"""
cubo_4_piezas.py

Cubo compuesto por 4 cubos 3D independientes.

- Visualmente: 4 cubos pequeños de distintos colores formando un "cubo principal".
- Cada cubo puede:
    * Permanecer "adjunto" al cubo principal (en su ranura).
    * Separarse y moverse individualmente.
    * Volver a colocarse en su ranura original.
- El cubo completo se puede mover como un bloque (solo las piezas adjuntas).

Además:
- Se define un cubo principal (main_index = 0).
- Cada pieza tiene rotación local (local_angles).
- Al recolocar una pieza separada, su rotación local se reinicia a 0,
  adaptándose a la rotación global del cubo principal.

Controles (solo para pruebas):
- W / A / S / D : mover TODO el conjunto (SOLO piezas adjuntas).
- 1 / 2 / 3 / 4 : seleccionar pieza (0,1,2,3).
- ESPACIO       : alternar adjuntar / separar pieza seleccionada.
- I / J / K / L : mover SOLO la pieza seleccionada (si está separada).
- Q             : salir.
"""

import cv2
import numpy as np
import math


# =====================================================
#  Render básico de cubo 3D
# =====================================================
class SimpleCube3D:
    """
    Renderiza un cubo 3D con proyección en perspectiva.
    """

    def __init__(self, color_bgr, base_size=1.0, distance=6.0):
        self.base_size = base_size
        self.distance = distance
        self.color = color_bgr

        s = self.base_size
        # Vértices de un cubo centrado en el origen
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
        Proyección perspectiva sencilla de puntos 3D al plano 2D.
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

    def draw(self, frame, scale, angle_x, angle_y, angle_z, offset_2d):
        """
        Dibuja el cubo sobre el frame.

        Devuelve:
        - pts_2d: puntos 2D proyectados (8 x 2).
        - bbox:  (min_x, max_x, min_y, max_y) en píxeles.
        """
        # 1) Escalar
        verts_scaled = self.base_vertices * scale

        # 2) Rotar
        R = self._rotation_matrix(angle_x, angle_y, angle_z)
        verts_rot = verts_scaled @ R.T

        # 3) Proyectar
        pts_2d, depths = self._project_points(verts_rot, frame.shape, f=800.0)

        # 4) Aplicar offset 2D
        if offset_2d is not None:
            pts_2d = pts_2d + offset_2d

        # 5) Bounding box
        min_x = float(np.min(pts_2d[:, 0]))
        max_x = float(np.max(pts_2d[:, 0]))
        min_y = float(np.min(pts_2d[:, 1]))
        max_y = float(np.max(pts_2d[:, 1]))
        bbox = (min_x, max_x, min_y, max_y)

        # 6) Ordenar caras por profundidad (pintar de atrás adelante)
        face_depths = []
        for i, face in enumerate(self.faces):
            z_mean = depths[face].mean()
            face_depths.append((z_mean, i))
        face_depths.sort(reverse=True)

        for _, face_idx in face_depths:
            face = self.faces[face_idx]
            color = self.color

            pts_face = pts_2d[face].astype(np.int32).reshape((-1, 1, 2))

            cv2.fillConvexPoly(frame, pts_face, color)
            cv2.polylines(frame, [pts_face], isClosed=True,
                          color=(255, 255, 255), thickness=1)

        return pts_2d, bbox


# =====================================================
#  Pieza individual del cubo compuesto
# =====================================================
class CubePiece:
    """
    Una pieza del cubo compuesto (un cubo pequeño).
    """

    def __init__(self, renderer, slot_offset_2d):
        self.renderer = renderer
        self.slot_offset_2d = np.array(slot_offset_2d, dtype=np.float32)

        # Estado de adjunción:
        self.attached = True
        # Offset cuando está separado
        self.free_offset_2d = None

        # Info para colisiones:
        self.last_pts_2d = None
        self.last_bbox = None

        # Flag para permitir primer movimiento sin colisión
        self.just_detached = False

        # Rotación local de la pieza (diferencial respecto a la global)
        # Se aplicará solo cuando la pieza esté separada.
        self.local_angles = np.zeros(3, dtype=np.float32)  # [ax, ay, az]

    def current_offset(self, group_offset_2d):
        """
        Offset actual (en píxeles) considerando si está adjunto o separado.
        """
        if self.attached:
            return group_offset_2d + self.slot_offset_2d
        else:
            return self.free_offset_2d

    def draw(self, frame, base_scale, base_angle_x, base_angle_y, base_angle_z,
             group_offset_2d):
        """
        Dibuja la pieza como cubo 3D, usando el offset y rotaciones adecuadas.
        """
        offset = self.current_offset(group_offset_2d)

        # Rotación total = rotación global + rotación local
        angle_x = base_angle_x + self.local_angles[0]
        angle_y = base_angle_y + self.local_angles[1]
        angle_z = base_angle_z + self.local_angles[2]

        pts_2d, bbox = self.renderer.draw(
            frame,
            scale=base_scale,
            angle_x=angle_x,
            angle_y=angle_y,
            angle_z=angle_z,
            offset_2d=offset
        )
        self.last_pts_2d = pts_2d
        self.last_bbox = bbox


# =====================================================
#  Cubo compuesto de 4 piezas
# =====================================================
class CompositeCubes:
    """
    Cubo principal formado por 4 cubos.

    - Todas las piezas comparten escala y rotación global.
    - Cada pieza puede tener una rotación local (cuando está separada).
    - Al recolocar una pieza separada, su rotación local se reinicia a 0,
      quedando alineada con la rotación global del cubo principal.
    """

    def __init__(self, initial_scale=0.3):
        self.scale = initial_scale
        self.angle_x = 0.0
        self.angle_y = 0.0
        self.angle_z = 0.0

        # Definimos qué pieza será el "cubo principal"
        self.main_index = 0

        # Offset 2D del grupo
        self.group_offset_2d = np.array([0.0, 0.0], dtype=np.float32)

        # Los 4 cubos están "pegados" en 2x2
        slot_gap = 80  # px aprox

        slots = [
            (-slot_gap / 2, -slot_gap / 2),  # arriba-izquierda (principal)
            ( slot_gap / 2, -slot_gap / 2),  # arriba-derecha
            (-slot_gap / 2,  slot_gap / 2),  # abajo-izquierda
            ( slot_gap / 2,  slot_gap / 2),  # abajo-derecha
        ]

        # Colores diferentes (BGR)
        colors = [
            (0, 0, 255),      # rojo
            (0, 255, 0),      # verde
            (255, 0, 0),      # azul
            (0, 255, 255),    # amarillo
        ]

        self.pieces = []
        for i in range(4):
            renderer = SimpleCube3D(color_bgr=colors[i], base_size=1.0, distance=6.0)
            piece = CubePiece(renderer, slot_offset_2d=slots[i])
            self.pieces.append(piece)

    # ---------- Dibujo ----------
    def draw(self, frame):
        for piece in self.pieces:
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
        Mueve TODO el conjunto (solo piezas ADJUNTAS).
        Las piezas separadas mantienen su posición independiente.
        """
        delta = np.array([dx, dy], dtype=np.float32)
        self.group_offset_2d += delta
        # Las piezas separadas NO se tocan.

    # ---------- Rotación global (para integrar con control de manos) ----------
    def rotate_global(self, d_ax, d_ay, d_az=0.0):
        """
        Rotación global del cubo principal (afecta a todas las piezas).
        """
        self.angle_x += d_ax
        self.angle_y += d_ay
        self.angle_z += d_az

    # ---------- Adjuntar/separar piezas ----------
    def toggle_attach(self, index):
        """
        Alterna entre adjunto / separado para la pieza 'index'.
        Al separar, la saca ligeramente hacia afuera para identificarla.
        Al recolocar, reinicia su rotación local para que se alinee con
        el cubo principal.
        """
        if not (0 <= index < len(self.pieces)):
            return

        piece = self.pieces[index]

        if piece.attached:
            # Separar la pieza -> su offset libre es su posición actual + pequeño desplazamiento
            base_offset = self.group_offset_2d + piece.slot_offset_2d

            # Dirección hacia afuera según su ranura
            dir_vec = piece.slot_offset_2d.copy()
            norm = np.linalg.norm(dir_vec)
            if norm < 1e-3:
                dir_vec = np.array([1.0, 0.0], dtype=np.float32)
            else:
                dir_vec = dir_vec / norm

            detach_dist = 80.0  # píxeles de separación inicial
            piece.free_offset_2d = base_offset + dir_vec * detach_dist

            piece.attached = False
            piece.just_detached = True  # primer movimiento sin colisión
            # Mantiene su rotación local (por si luego la giramos)
            print(f"[PIEZA {index}] Separada del cubo principal y desplazada ligeramente.")
        else:
            # Recolocar la pieza en su ranura actual del grupo
            piece.attached = True
            piece.free_offset_2d = None
            piece.just_detached = False
            # IMPORTANTE: al volver a unirse, se alinea con el cubo principal
            piece.local_angles[:] = 0.0
            print(f"[PIEZA {index}] Recolocada en el cubo principal (rotación alineada).")

    # ---------- Colisiones 2D ----------
    @staticmethod
    def _boxes_intersect(b1, b2):
        """
        Verifica intersección de bounding boxes 2D.
        b = (min_x, max_x, min_y, max_y)
        """
        x1_min, x1_max, y1_min, y1_max = b1
        x2_min, x2_max, y2_min, y2_max = b2

        if x1_max <= x2_min or x1_min >= x2_max:
            return False
        if y1_max <= y2_min or y1_min >= y2_max:
            return False
        return True

    # ---------- Movimiento de pieza individual ----------
    def move_piece(self, index, dx, dy):
        """
        Mueve una pieza separada en 2D, evitando encimarse con otras piezas.
        Si la pieza se acaba de separar (just_detached), se permite el primer
        movimiento sin comprobar colisión.
        """
        if not (0 <= index < len(self.pieces)):
            return

        piece = self.pieces[index]

        if piece.attached:
            # Si está adjunta, no se mueve individualmente
            return

        if piece.free_offset_2d is None:
            piece.free_offset_2d = self.group_offset_2d.copy()

        delta = np.array([dx, dy], dtype=np.float32)

        # Primer movimiento tras separarse: sin colisión
        if piece.last_bbox is None or piece.just_detached:
            piece.free_offset_2d += delta
            piece.just_detached = False
            return

        # Bbox candidata tras el movimiento
        x_min, x_max, y_min, y_max = piece.last_bbox
        cand_bbox = (
            x_min + dx,
            x_max + dx,
            y_min + dy,
            y_max + dy
        )

        # Comprobar colisiones con las demás piezas
        for j, other in enumerate(self.pieces):
            if j == index:
                continue
            if other.last_bbox is None:
                continue

            if self._boxes_intersect(cand_bbox, other.last_bbox):
                print(f"[PIEZA {index}] Movimiento bloqueado por colisión con pieza {j}.")
                return

        # Si no hay colisión, aplicar desplazamiento
        piece.free_offset_2d += delta


# =====================================================
#  main() de prueba
# =====================================================
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return

    composite = CompositeCubes(initial_scale=0.3)

    print("Cámara + cubo compuesto de 4 piezas.")
    print("Controles (SOLO para pruebas):")
    print("  W/A/S/D : mover TODO el conjunto (solo piezas adjuntas).")
    print("  1/2/3/4 : seleccionar pieza (0..3).")
    print("  ESPACIO : separar / recolocar la pieza seleccionada.")
    print("  I/J/K/L : mover pieza seleccionada (si está separada).")
    print("  Q       : salir.")

    selected_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        composite.draw(frame)

        cv2.putText(
            frame,
            f"Pieza seleccionada: {selected_index}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.imshow("Cubo compuesto (4 piezas)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q')):
            break

        # Selección de pieza
        if key in (ord('1'), ord('2'), ord('3'), ord('4')):
            selected_index = key - ord('1')
            print(f"[SELECCIÓN] Pieza actual: {selected_index}")

        # Mover grupo completo (solo piezas adjuntas)
        step_group = 10
        if key in (ord('w'), ord('W')):
            composite.move_group(0, -step_group)
        elif key in (ord('s'), ord('S')):
            composite.move_group(0, step_group)
        elif key in (ord('a'), ord('A')):
            composite.move_group(-step_group, 0)
        elif key in (ord('d'), ord('D')):
            composite.move_group(step_group, 0)

        # Separar / recolocar pieza seleccionada
        if key == ord(' '):
            composite.toggle_attach(selected_index)

        # Mover solo la pieza seleccionada (si está separada)
        step_piece = 10
        if key in (ord('i'), ord('I')):
            composite.move_piece(selected_index, 0, -step_piece)
        elif key in (ord('k'), ord('K')):
            composite.move_piece(selected_index, 0, step_piece)
        elif key in (ord('j'), ord('J')):
            composite.move_piece(selected_index, -step_piece, 0)
        elif key in (ord('l'), ord('L')):
            composite.move_piece(selected_index, step_piece, 0)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

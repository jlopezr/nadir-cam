#!/usr/bin/env python3
"""
nadir_cam_rpy.py

Cámara virtual desde WMS usando pose completa de cámara:
- lat, lon, alt
- roll, pitch, yaw

Suposiciones:
- terreno plano: z_ground = 0
- altitud es AGL en metros
- cámara y dron están en el mismo punto
- los ángulos son los de la cámara
- la cámara en pose neutra mira hacia abajo
- marco mundo local ENU: x=este, y=norte, z=arriba
- yaw: 0=norte, positivo horario

Ejemplo:
python nadir_cam_rpy.py \
  --wms-url "https://www.ign.es/wms-inspire/pnoa-ma" \
  --layer "OI.OrthoimageCoverage" \
  --lat 40.4168 \
  --lon -3.7038 \
  --alt 120 \
  --roll 0 \
  --pitch 0 \
  --yaw 30 \
  --hfov 70 \
  --width 1280 \
  --height 720 \
  --out frame.png
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from io import BytesIO
from typing import Tuple

import cv2
import numpy as np
import requests
from PIL import Image
from pyproj import Transformer


@dataclass
class CameraSpec:
    width_px: int
    height_px: int
    hfov_deg: float

    @property
    def aspect_ratio(self) -> float:
        return self.width_px / self.height_px

    @property
    def hfov_rad(self) -> float:
        return math.radians(self.hfov_deg)

    @property
    def vfov_rad(self) -> float:
        return 2.0 * math.atan(math.tan(self.hfov_rad / 2.0) / self.aspect_ratio)

    @property
    def fx(self) -> float:
        return self.width_px / (2.0 * math.tan(self.hfov_rad / 2.0))

    @property
    def fy(self) -> float:
        return self.fx  # asumimos píxel cuadrado

    @property
    def cx(self) -> float:
        return self.width_px / 2.0

    @property
    def cy(self) -> float:
        return self.height_px / 2.0


@dataclass
class Pose:
    lat_deg: float
    lon_deg: float
    alt_agl_m: float
    roll_deg: float
    pitch_deg: float
    yaw_deg: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate camera image from WMS using camera roll/pitch/yaw.")
    p.add_argument("--wms-url", required=True)
    p.add_argument("--layer", required=True)
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True)
    p.add_argument("--alt", type=float, required=True, help="Altitude AGL in meters")
    p.add_argument("--roll", type=float, default=0.0)
    p.add_argument("--pitch", type=float, default=0.0)
    p.add_argument("--yaw", type=float, default=0.0, help="0=north, clockwise positive")
    p.add_argument("--hfov", type=float, required=True)
    p.add_argument("--width", type=int, required=True)
    p.add_argument("--height", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--wms-version", default="1.3.0")
    p.add_argument("--format", default="image/png")
    p.add_argument("--timeout", type=float, default=20.0)
    p.add_argument("--margin", type=float, default=1.15, help="Extra bbox margin factor")
    return p.parse_args()


def latlon_to_webmercator(lat_deg: float, lon_deg: float) -> Tuple[float, float]:
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x_m, y_m = transformer.transform(lon_deg, lat_deg)
    return x_m, y_m


def fetch_wms_image(
    wms_url: str,
    layer: str,
    bbox_3857: Tuple[float, float, float, float],
    width_px: int,
    height_px: int,
    image_format: str,
    version: str,
    timeout_s: float,
) -> Image.Image:
    minx, miny, maxx, maxy = bbox_3857

    params = {
        "SERVICE": "WMS",
        "REQUEST": "GetMap",
        "VERSION": version,
        "LAYERS": layer,
        "STYLES": "",
        "FORMAT": image_format,
        "TRANSPARENT": "FALSE",
        "WIDTH": str(width_px),
        "HEIGHT": str(height_px),
    }

    if version == "1.3.0":
        params["CRS"] = "EPSG:3857"
        params["BBOX"] = f"{minx},{miny},{maxx},{maxy}"
    else:
        params["SRS"] = "EPSG:3857"
        params["BBOX"] = f"{minx},{miny},{maxx},{maxy}"

    r = requests.get(wms_url, params=params, timeout=timeout_s)
    r.raise_for_status()

    content_type = r.headers.get("Content-Type", "")
    if "image" not in content_type.lower():
        raise RuntimeError(
            f"El servidor WMS no devolvió imagen. Content-Type={content_type!r}. "
            f"Respuesta={r.text[:300]!r}"
        )

    return Image.open(BytesIO(r.content)).convert("RGB")


def pil_to_cv(image: Image.Image) -> np.ndarray:
    arr = np.array(image)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def cv_to_pil(image: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c],
    ], dtype=np.float64)


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c],
    ], dtype=np.float64)


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1],
    ], dtype=np.float64)


def camera_to_world_rotation(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """
    Construye R_cw: vector en cámara -> vector en mundo ENU.

    Cámara base:
    - +x: derecha en imagen
    - +y: abajo en imagen
    - +z: hacia delante (óptico)

    En pose neutra la cámara mira hacia abajo:
    x_cam -> +Este
    y_cam -> -Norte
    z_cam -> -Arriba

    Luego aplicamos:
    - yaw alrededor de +Z mundo, pero como el yaw de entrada es horario desde norte,
      lo convertimos a ángulo matemático ENU.
    - pitch alrededor del eje X local de cámara
    - roll alrededor del eje Z local de cámara

    Esta convención es una aproximación útil para arrancar; puede requerir
    ajuste de signos según tu pipeline real.
    """
    # Base camera->world when looking straight down.
    r_base = np.array([
        [1,  0,  0],   # x_cam -> +E
        [0, -1,  0],   # y_cam -> -N
        [0,  0, -1],   # z_cam -> -U
    ], dtype=np.float64)

    # yaw entrada: 0=norte, horario positivo
    # yaw matemático ENU: 0=este, antihorario positivo
    yaw_math = math.radians(90.0 - yaw_deg)

    # Rotaciones "externas" en mundo para heading
    r_yaw_world = rot_z(yaw_math)

    # Rotaciones "internas" aproximadas de cámara
    r_roll_cam = rot_z(math.radians(roll_deg))
    r_pitch_cam = rot_x(math.radians(pitch_deg))

    return r_yaw_world @ r_base @ r_pitch_cam @ r_roll_cam


def pixel_to_camera_ray(u: float, v: float, cam: CameraSpec) -> np.ndarray:
    x = (u - cam.cx) / cam.fx
    y = (v - cam.cy) / cam.fy
    ray = np.array([x, y, 1.0], dtype=np.float64)
    return ray / np.linalg.norm(ray)


def intersect_ray_with_ground(
    cam_pos_world: np.ndarray,
    ray_world: np.ndarray,
    z_ground: float = 0.0,
) -> np.ndarray:
    """
    Intersección de un rayo P = cam_pos + t * ray con el plano z=z_ground.
    """
    dz = ray_world[2]
    if abs(dz) < 1e-9:
        raise ValueError("Rayo casi paralelo al suelo")

    t = (z_ground - cam_pos_world[2]) / dz
    if t <= 0:
        raise ValueError("El rayo no intersecta el suelo delante de la cámara")

    return cam_pos_world + t * ray_world


def world_to_bbox(points_xy: np.ndarray, margin: float) -> Tuple[float, float, float, float]:
    minx = float(points_xy[:, 0].min())
    maxx = float(points_xy[:, 0].max())
    miny = float(points_xy[:, 1].min())
    maxy = float(points_xy[:, 1].max())

    cx = 0.5 * (minx + maxx)
    cy = 0.5 * (miny + maxy)
    w = (maxx - minx) * margin
    h = (maxy - miny) * margin

    return (
        cx - 0.5 * w,
        cy - 0.5 * h,
        cx + 0.5 * w,
        cy + 0.5 * h,
    )


def world_xy_to_image_px(
    points_xy: np.ndarray,
    bbox_3857: Tuple[float, float, float, float],
    img_w: int,
    img_h: int,
) -> np.ndarray:
    minx, miny, maxx, maxy = bbox_3857

    xs = points_xy[:, 0]
    ys = points_xy[:, 1]

    us = (xs - minx) / (maxx - minx) * img_w
    vs = (maxy - ys) / (maxy - miny) * img_h  # y imagen hacia abajo

    return np.stack([us, vs], axis=1).astype(np.float32)


def main() -> int:
    args = parse_args()

    if args.alt <= 0:
        raise ValueError("La altitud debe ser > 0")
    if not (0 < args.hfov < 179):
        raise ValueError("hfov debe estar entre 0 y 179 grados")

    cam = CameraSpec(
        width_px=args.width,
        height_px=args.height,
        hfov_deg=args.hfov,
    )
    pose = Pose(
        lat_deg=args.lat,
        lon_deg=args.lon,
        alt_agl_m=args.alt,
        roll_deg=args.roll,
        pitch_deg=args.pitch,
        yaw_deg=args.yaw,
    )

    center_x, center_y = latlon_to_webmercator(pose.lat_deg, pose.lon_deg)

    cam_pos_world = np.array([center_x, center_y, pose.alt_agl_m], dtype=np.float64)
    r_cw = camera_to_world_rotation(pose.roll_deg, pose.pitch_deg, pose.yaw_deg)

    # corners de imagen
    img_corners_uv = np.array([
        [0, 0],
        [cam.width_px - 1, 0],
        [cam.width_px - 1, cam.height_px - 1],
        [0, cam.height_px - 1],
    ], dtype=np.float64)

    ground_pts = []
    for u, v in img_corners_uv:
        ray_cam = pixel_to_camera_ray(u, v, cam)
        ray_world = r_cw @ ray_cam
        p_ground = intersect_ray_with_ground(cam_pos_world, ray_world, z_ground=0.0)
        ground_pts.append(p_ground[:2])

    ground_pts = np.array(ground_pts, dtype=np.float64)

    bbox_3857 = world_to_bbox(ground_pts, margin=args.margin)

    wms_img = fetch_wms_image(
        wms_url=args.wms_url,
        layer=args.layer,
        bbox_3857=bbox_3857,
        width_px=cam.width_px,
        height_px=cam.height_px,
        image_format=args.format,
        version=args.wms_version,
        timeout_s=args.timeout,
    )

    src_bgr = pil_to_cv(wms_img)

    src_pts = world_xy_to_image_px(
        ground_pts,
        bbox_3857=bbox_3857,
        img_w=cam.width_px,
        img_h=cam.height_px,
    )

    dst_pts = np.array([
        [0, 0],
        [cam.width_px - 1, 0],
        [cam.width_px - 1, cam.height_px - 1],
        [0, cam.height_px - 1],
    ], dtype=np.float32)

    h_mat = cv2.getPerspectiveTransform(src_pts, dst_pts)

    out_bgr = cv2.warpPerspective(
        src_bgr,
        h_mat,
        (cam.width_px, cam.height_px),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    out_img = cv_to_pil(out_bgr)
    out_img.save(args.out)

    print(f"Imagen guardada en: {args.out}")
    print("Ground corners (EPSG:3857):")
    for i, p in enumerate(ground_pts):
        print(f"  corner_{i}: x={p[0]:.3f}, y={p[1]:.3f}")
    print(f"BBOX solicitado: {bbox_3857}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
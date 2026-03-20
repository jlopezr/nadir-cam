#!/usr/bin/env python3
"""
nadir_cam.py

MVP de cámara nadir sintética desde WMS.

Suposiciones de esta v1:
- cámara siempre apuntando hacia abajo
- terreno plano
- altitud en metros AGL
- yaw en grados, 0 = norte, positivo en sentido horario
- el WMS soporta EPSG:3857 y formato image/png

Ejemplo:
python nadir_cam.py \
  --wms-url "https://www.ign.es/wms-inspire/pnoa-ma" \
  --layer "OI.OrthoimageCoverage" \
  --lat 40.4168 \
  --lon -3.7038 \
  --alt 120 \
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


EARTH_RADIUS_M = 6378137.0


@dataclass
class CameraSpec:
    width_px: int
    height_px: int
    hfov_deg: float

    @property
    def aspect_ratio(self) -> float:
        return self.width_px / self.height_px

    @property
    def vfov_rad(self) -> float:
        hfov_rad = math.radians(self.hfov_deg)
        return 2.0 * math.atan(math.tan(hfov_rad / 2.0) / self.aspect_ratio)

    @property
    def hfov_rad(self) -> float:
        return math.radians(self.hfov_deg)


@dataclass
class Pose:
    lat_deg: float
    lon_deg: float
    alt_agl_m: float
    yaw_deg: float


@dataclass
class GroundFootprint:
    width_m: float
    height_m: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate nadir camera image from WMS.")
    p.add_argument("--wms-url", required=True, help="Base WMS URL")
    p.add_argument("--layer", required=True, help="WMS layer name")
    p.add_argument("--lat", type=float, required=True, help="Latitude in degrees")
    p.add_argument("--lon", type=float, required=True, help="Longitude in degrees")
    p.add_argument("--alt", type=float, required=True, help="Altitude AGL in meters")
    p.add_argument("--yaw", type=float, default=0.0, help="Yaw in degrees, 0=north, clockwise positive")
    p.add_argument("--hfov", type=float, required=True, help="Horizontal FOV in degrees")
    p.add_argument("--width", type=int, required=True, help="Output width in pixels")
    p.add_argument("--height", type=int, required=True, help="Output height in pixels")
    p.add_argument("--out", required=True, help="Output image path")
    p.add_argument("--wms-version", default="1.3.0", help="WMS version, default 1.3.0")
    p.add_argument("--format", default="image/png", help="Requested image format")
    p.add_argument("--oversample", type=float, default=1.6, help="Extra margin before rotation/crop")
    p.add_argument("--timeout", type=float, default=20.0, help="HTTP timeout in seconds")
    return p.parse_args()


def compute_ground_footprint(camera: CameraSpec, alt_agl_m: float) -> GroundFootprint:
    width_m = 2.0 * alt_agl_m * math.tan(camera.hfov_rad / 2.0)
    height_m = 2.0 * alt_agl_m * math.tan(camera.vfov_rad / 2.0)
    return GroundFootprint(width_m=width_m, height_m=height_m)


def latlon_to_webmercator(lat_deg: float, lon_deg: float) -> Tuple[float, float]:
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x_m, y_m = transformer.transform(lon_deg, lat_deg)
    return x_m, y_m


def build_bbox_3857(center_x_m: float, center_y_m: float, width_m: float, height_m: float) -> Tuple[float, float, float, float]:
    half_w = width_m / 2.0
    half_h = height_m / 2.0
    return (
        center_x_m - half_w,  # minx
        center_y_m - half_h,  # miny
        center_x_m + half_w,  # maxx
        center_y_m + half_h,  # maxy
    )


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

    # Para EPSG:3857 el orden del bbox no suele dar guerra, pero mantenemos
    # la diferencia entre WMS 1.1.1 y 1.3.0 por limpieza.
    if version == "1.3.0":
        params["CRS"] = "EPSG:3857"
        params["BBOX"] = f"{minx},{miny},{maxx},{maxy}"
    else:
        params["SRS"] = "EPSG:3857"
        params["BBOX"] = f"{minx},{miny},{maxx},{maxy}"

    response = requests.get(wms_url, params=params, timeout=timeout_s)
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "")
    if "image" not in content_type.lower():
        raise RuntimeError(
            f"El servidor WMS no devolvió una imagen. Content-Type={content_type!r}. "
            f"Primeros bytes: {response.text[:300]!r}"
        )

    return Image.open(BytesIO(response.content)).convert("RGB")


def pil_to_cv(image: Image.Image) -> np.ndarray:
    arr = np.array(image)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def cv_to_pil(image: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def rotate_image_for_yaw(image_bgr: np.ndarray, yaw_deg: float) -> np.ndarray:
    """
    Convención:
    - yaw = 0 => norte arriba, sin rotación extra
    - yaw positivo horario
    - la imagen de OpenCV rota positivo antihorario
    """
    h, w = image_bgr.shape[:2]
    center = (w / 2.0, h / 2.0)

    # Para que la escena gire como una cámara con heading horario,
    # aplicamos el signo contrario en OpenCV.
    rot_mat = cv2.getRotationMatrix2D(center, -yaw_deg, 1.0)

    cos = abs(rot_mat[0, 0])
    sin = abs(rot_mat[0, 1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    rot_mat[0, 2] += (new_w / 2.0) - center[0]
    rot_mat[1, 2] += (new_h / 2.0) - center[1]

    return cv2.warpAffine(
        image_bgr,
        rot_mat,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )


def center_crop(image_bgr: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    if out_w > w or out_h > h:
        raise ValueError(f"Crop {out_w}x{out_h} no cabe en imagen {w}x{h}")

    x0 = (w - out_w) // 2
    y0 = (h - out_h) // 2
    return image_bgr[y0:y0 + out_h, x0:x0 + out_w]


def main() -> int:
    args = parse_args()

    camera = CameraSpec(
        width_px=args.width,
        height_px=args.height,
        hfov_deg=args.hfov,
    )
    pose = Pose(
        lat_deg=args.lat,
        lon_deg=args.lon,
        alt_agl_m=args.alt,
        yaw_deg=args.yaw,
    )

    if pose.alt_agl_m <= 0:
        raise ValueError("La altitud debe ser > 0 m")
    if not (0 < camera.hfov_deg < 179):
        raise ValueError("hfov debe estar entre 0 y 179 grados")

    footprint = compute_ground_footprint(camera, pose.alt_agl_m)

    # Pedimos al WMS una imagen un poco más grande para poder rotar y recortar.
    req_ground_w = footprint.width_m * args.oversample
    req_ground_h = footprint.height_m * args.oversample
    req_px_w = int(camera.width_px * args.oversample)
    req_px_h = int(camera.height_px * args.oversample)

    center_x_m, center_y_m = latlon_to_webmercator(pose.lat_deg, pose.lon_deg)
    bbox_3857 = build_bbox_3857(center_x_m, center_y_m, req_ground_w, req_ground_h)

    wms_img = fetch_wms_image(
        wms_url=args.wms_url,
        layer=args.layer,
        bbox_3857=bbox_3857,
        width_px=req_px_w,
        height_px=req_px_h,
        image_format=args.format,
        version=args.wms_version,
        timeout_s=args.timeout,
    )

    img_bgr = pil_to_cv(wms_img)
    rotated_bgr = rotate_image_for_yaw(img_bgr, pose.yaw_deg)
    cropped_bgr = center_crop(rotated_bgr, camera.width_px, camera.height_px)

    out_img = cv_to_pil(cropped_bgr)
    out_img.save(args.out)

    print(f"Imagen guardada en: {args.out}")
    print(f"Footprint suelo aprox: {footprint.width_m:.2f} m x {footprint.height_m:.2f} m")
    print(f"BBOX solicitado EPSG:3857: {bbox_3857}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
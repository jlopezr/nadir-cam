#!/usr/bin/env python3
"""
terrain_cam_wcs.py

Genera una imagen de cámara virtual usando:
- ortofoto por WMS
- DEM por WCS

Modelo:
- la altitud de entrada es AGL
- el DEM aporta el relieve
- la cámara usa convención aeronáutica clásica roll/pitch/yaw
- mundo local ENU: x=este, y=norte, z=arriba
- yaw: 0=norte, positivo horario

Notas:
- la ortofoto se pide en EPSG:3857
- el DEM se pide en EPSG:3857
- el script asume que el WCS devuelve un GeoTIFF legible por rasterio
- el render ya no usa homografía: hace intersección rayo-terreno por píxel
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from io import BytesIO
from typing import Tuple

import numpy as np
import requests
import rasterio
from PIL import Image
from pyproj import Transformer
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling


# ---------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------

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
        return self.fx  # píxel cuadrado

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


# ---------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate camera image from WMS + DEM(WCS) using roll/pitch/yaw.")
    p.add_argument("--ortho-wms-url", required=True)
    p.add_argument("--ortho-layer", required=True)

    p.add_argument("--dem-wcs-url", required=True)
    p.add_argument("--dem-coverage", required=True)

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

    p.add_argument("--ortho-wms-version", default="1.3.0")
    p.add_argument("--ortho-format", default="image/png")

    p.add_argument("--dem-wcs-version", default="2.0.1")
    p.add_argument("--dem-format", default="image/tiff")

    p.add_argument("--timeout", type=float, default=30.0)

    p.add_argument("--src-width", type=int, default=2048, help="Source ortho/DEM width")
    p.add_argument("--src-height", type=int, default=2048, help="Source ortho/DEM height")

    p.add_argument("--margin", type=float, default=1.35, help="Extra bbox scale factor")
    p.add_argument("--terrain-extra-m", type=float, default=100.0, help="Extra bbox padding in meters")

    p.add_argument("--march-steps", type=int, default=64, help="Ray marching steps")
    p.add_argument("--bisection-steps", type=int, default=12, help="Bisection refinement steps")
    p.add_argument("--chunk-size", type=int, default=20000, help="Pixels per chunk")
    return p.parse_args()


# ---------------------------------------------------------------------
# CRS helpers
# ---------------------------------------------------------------------

def latlon_to_webmercator(lat_deg: float, lon_deg: float) -> Tuple[float, float]:
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x_m, y_m = transformer.transform(lon_deg, lat_deg)
    return x_m, y_m


# ---------------------------------------------------------------------
# Fetch ortho / DEM
# ---------------------------------------------------------------------

def fetch_wms_image(
    wms_url: str,
    layer: str,
    bbox_3857: Tuple[float, float, float, float],
    width_px: int,
    height_px: int,
    image_format: str,
    version: str,
    timeout_s: float,
) -> tuple[np.ndarray, rasterio.Affine]:
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

    img = Image.open(BytesIO(r.content)).convert("RGB")
    arr = np.array(img, dtype=np.uint8)
    transform = from_bounds(minx, miny, maxx, maxy, width_px, height_px)
    return arr, transform

def fetch_wcs_dem(
    wcs_url: str,
    coverage: str,
    bbox_3857: Tuple[float, float, float, float],
    width_px: int,
    height_px: int,
    fmt: str,
    version: str,
    timeout_s: float,
) -> tuple[np.ndarray, rasterio.Affine]:
    """
    Descarga un DEM del IGN por WCS 2.0.1 y lo reproyecta a EPSG:3857.

    Así el resto del pipeline puede seguir trabajando íntegramente en 3857.
    """

    if version != "2.0.1":
        raise ValueError("Esta implementación está preparada para WCS 2.0.1 del IGN")

    minx, miny, maxx, maxy = bbox_3857

    # CRS nativo según CoverageId del IGN
    if coverage.startswith("Elevacion4258_"):
        coverage_crs = "EPSG:4258"
    elif coverage.startswith("Elevacion25830_"):
        coverage_crs = "EPSG:25830"
    else:
        raise ValueError(
            f"No sé qué CRS usa la cobertura {coverage!r}. "
            "Este código soporta Elevacion4258_* y Elevacion25830_*."
        )

    # Transformar el bbox solicitado (3857) al CRS nativo de la cobertura
    transformer = Transformer.from_crs("EPSG:3857", coverage_crs, always_xy=True)
    x1, y1 = transformer.transform(minx, miny)
    x2, y2 = transformer.transform(maxx, maxy)

    cov_minx, cov_maxx = sorted((x1, x2))
    cov_miny, cov_maxy = sorted((y1, y2))

    params = [
        ("SERVICE", "WCS"),
        ("REQUEST", "GetCoverage"),
        ("VERSION", "2.0.1"),
        ("COVERAGEID", coverage),
        ("FORMAT", fmt),
        ("SUBSET", f"x({cov_minx},{cov_maxx})"),
        ("SUBSET", f"y({cov_miny},{cov_maxy})"),
        ("interpolationMethod", "bilinear"),
    ]

    r = requests.get(wcs_url, params=params, timeout=timeout_s)
    r.raise_for_status()

    content_type = r.headers.get("Content-Type", "").lower()
    if not any(x in content_type for x in ["image/tiff", "image/geotiff", "geotiff", "tiff", "application/octet-stream"]):
        raise RuntimeError(
            f"El servidor WCS no devolvió un GeoTIFF reconocible. "
            f"Content-Type={content_type!r}. Respuesta={r.text[:500]!r}"
        )

    with MemoryFile(r.content) as memfile:
        with memfile.open() as ds:
            src_dem = ds.read(1).astype(np.float64)
            src_nodata = ds.nodata
            if src_nodata is not None:
                src_dem[src_dem == src_nodata] = np.nan

            src_transform = ds.transform
            src_crs = ds.crs

            if src_crs is None:
                raise RuntimeError("El GeoTIFF del WCS no incluye CRS")

            # Reproyectar a 3857 para que el resto del script siga igual
            dst_transform = from_bounds(minx, miny, maxx, maxy, width_px, height_px)
            dst_dem = np.full((height_px, width_px), np.nan, dtype=np.float64)

            reproject(
                source=src_dem,
                destination=dst_dem,
                src_transform=src_transform,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs="EPSG:3857",
                src_nodata=np.nan,
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )

    return dst_dem, dst_transform

# ---------------------------------------------------------------------
# Rotations
# ---------------------------------------------------------------------

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
    R_cw: vector en cámara -> vector en mundo ENU
    usando convención aeronáutica clásica:

    - roll:  alrededor del eje longitudinal
    - pitch: alrededor del eje lateral
    - yaw:   alrededor del eje vertical
    - yaw = 0 -> norte
    - yaw positivo -> horario

    Cámara:
    - x_cam = derecha en imagen
    - y_cam = abajo en imagen
    - z_cam = eje óptico, hacia delante

    Pose neutra:
    - cámara mirando hacia abajo
    - parte superior de la imagen = norte
    - derecha de la imagen = este
    """
    roll = math.radians(roll_deg)
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)

    # camera -> body aeronáutico
    # x_body (forward) = -y_cam
    # y_body (right)   = +x_cam
    # z_body (down)    = +z_cam
    r_bc = np.array([
        [0, -1,  0],
        [1,  0,  0],
        [0,  0,  1],
    ], dtype=np.float64)

    # body -> NED
    r_nb = rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)

    # NED -> ENU
    r_en = np.array([
        [0,  1,  0],
        [1,  0,  0],
        [0,  0, -1],
    ], dtype=np.float64)

    return r_en @ r_nb @ r_bc


# ---------------------------------------------------------------------
# Camera / rays
# ---------------------------------------------------------------------

def pixel_to_camera_ray(u: np.ndarray, v: np.ndarray, cam: CameraSpec) -> np.ndarray:
    x = (u - cam.cx) / cam.fx
    y = (v - cam.cy) / cam.fy
    z = np.ones_like(x, dtype=np.float64)
    rays = np.stack([x, y, z], axis=1)
    norms = np.linalg.norm(rays, axis=1, keepdims=True)
    return rays / norms


# ---------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------

def world_to_frac_rc(transform: rasterio.Affine, xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    inv = ~transform
    cols, rows = inv * (xs, ys)
    return rows, cols


def bilinear_sample_scalar(
    data: np.ndarray,
    transform: rasterio.Affine,
    xs: np.ndarray,
    ys: np.ndarray,
    fill_value: float = np.nan,
) -> np.ndarray:
    rows, cols = world_to_frac_rc(transform, xs, ys)

    r0 = np.floor(rows).astype(np.int64)
    c0 = np.floor(cols).astype(np.int64)
    r1 = r0 + 1
    c1 = c0 + 1

    h, w = data.shape
    valid = (r0 >= 0) & (c0 >= 0) & (r1 < h) & (c1 < w)

    out = np.full(xs.shape, fill_value, dtype=np.float64)
    if not np.any(valid):
        return out

    rr0 = r0[valid]
    cc0 = c0[valid]
    rr1 = r1[valid]
    cc1 = c1[valid]

    dr = rows[valid] - rr0
    dc = cols[valid] - cc0

    q00 = data[rr0, cc0]
    q10 = data[rr1, cc0]
    q01 = data[rr0, cc1]
    q11 = data[rr1, cc1]

    finite = np.isfinite(q00) & np.isfinite(q10) & np.isfinite(q01) & np.isfinite(q11)
    if np.any(finite):
        drf = dr[finite]
        dcf = dc[finite]
        vals = (
            q00[finite] * (1.0 - drf) * (1.0 - dcf) +
            q10[finite] * drf * (1.0 - dcf) +
            q01[finite] * (1.0 - drf) * dcf +
            q11[finite] * drf * dcf
        )
        out_valid = out[valid]
        out_valid[finite] = vals
        out[valid] = out_valid

    return out


def bilinear_sample_rgb(
    data: np.ndarray,
    transform: rasterio.Affine,
    xs: np.ndarray,
    ys: np.ndarray,
) -> np.ndarray:
    rows, cols = world_to_frac_rc(transform, xs, ys)

    r0 = np.floor(rows).astype(np.int64)
    c0 = np.floor(cols).astype(np.int64)
    r1 = r0 + 1
    c1 = c0 + 1

    h, w, _ = data.shape
    valid = (r0 >= 0) & (c0 >= 0) & (r1 < h) & (c1 < w)

    out = np.zeros((xs.shape[0], 3), dtype=np.float64)
    if not np.any(valid):
        return out.astype(np.uint8)

    rr0 = r0[valid]
    cc0 = c0[valid]
    rr1 = r1[valid]
    cc1 = c1[valid]

    dr = (rows[valid] - rr0)[:, None]
    dc = (cols[valid] - cc0)[:, None]

    q00 = data[rr0, cc0].astype(np.float64)
    q10 = data[rr1, cc0].astype(np.float64)
    q01 = data[rr0, cc1].astype(np.float64)
    q11 = data[rr1, cc1].astype(np.float64)

    vals = (
        q00 * (1.0 - dr) * (1.0 - dc) +
        q10 * dr * (1.0 - dc) +
        q01 * (1.0 - dr) * dc +
        q11 * dr * dc
    )

    out[valid] = vals
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------
# BBOX estimation
# ---------------------------------------------------------------------

def estimate_fetch_bbox(
    center_x: float,
    center_y: float,
    cam: CameraSpec,
    pose: Pose,
    margin: float,
    terrain_extra_m: float,
) -> Tuple[float, float, float, float]:
    """
    Estima un bbox suficientemente grande usando la huella aproximada
    sobre un plano local z=0 solo para decidir qué DEM/ortofoto pedir.
    """
    r_cw = camera_to_world_rotation(pose.roll_deg, pose.pitch_deg, pose.yaw_deg)

    sample_uv = np.array([
        [0.5, 0.5],
        [cam.width_px - 0.5, 0.5],
        [cam.width_px - 0.5, cam.height_px - 0.5],
        [0.5, cam.height_px - 0.5],
        [cam.cx, cam.cy],
        [cam.cx, 0.5],
        [cam.cx, cam.height_px - 0.5],
        [0.5, cam.cy],
        [cam.width_px - 0.5, cam.cy],
    ], dtype=np.float64)

    pts = []
    for u, v in sample_uv:
        ray_cam = pixel_to_camera_ray(np.array([u]), np.array([v]), cam)[0]
        ray_world = r_cw @ ray_cam

        # Para estimar el área a descargar, usamos z=0 local.
        # Si el rayo sale casi horizontal o hacia arriba, forzamos una distancia grande.
        if ray_world[2] < -1e-3:
            t = pose.alt_agl_m / (-ray_world[2])
        else:
            t = pose.alt_agl_m / 0.05

        x = t * ray_world[0]
        y = t * ray_world[1]
        pts.append([x, y])

    pts = np.array(pts, dtype=np.float64)
    pts = np.vstack([pts, np.array([[0.0, 0.0]], dtype=np.float64)])

    minx = float(pts[:, 0].min())
    maxx = float(pts[:, 0].max())
    miny = float(pts[:, 1].min())
    maxy = float(pts[:, 1].max())

    cx = 0.5 * (minx + maxx)
    cy = 0.5 * (miny + maxy)
    w = (maxx - minx) * margin + 2.0 * terrain_extra_m
    h = (maxy - miny) * margin + 2.0 * terrain_extra_m

    return (
        center_x + cx - 0.5 * w,
        center_y + cy - 0.5 * h,
        center_x + cx + 0.5 * w,
        center_y + cy + 0.5 * h,
    )


# ---------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------

def render_virtual_view(
    cam: CameraSpec,
    pose: Pose,
    center_x: float,
    center_y: float,
    bbox_3857: Tuple[float, float, float, float],
    dem: np.ndarray,
    dem_transform: rasterio.Affine,
    ortho_rgb: np.ndarray,
    ortho_transform: rasterio.Affine,
    center_ground_z: float,
    march_steps: int,
    bisection_steps: int,
    chunk_size: int,
) -> np.ndarray:
    out = np.zeros((cam.height_px, cam.width_px, 3), dtype=np.uint8)

    cam_z = center_ground_z + pose.alt_agl_m
    r_cw = camera_to_world_rotation(pose.roll_deg, pose.pitch_deg, pose.yaw_deg)

    minx, miny, maxx, maxy = bbox_3857
    rel_minx = minx - center_x
    rel_maxx = maxx - center_x
    rel_miny = miny - center_y
    rel_maxy = maxy - center_y

    flat_out = out.reshape(-1, 3)
    total_px = cam.width_px * cam.height_px

    for start in range(0, total_px, chunk_size):
        end = min(start + chunk_size, total_px)
        idx = np.arange(start, end, dtype=np.int64)

        v = idx // cam.width_px
        u = idx % cam.width_px

        u_f = u.astype(np.float64) + 0.5
        v_f = v.astype(np.float64) + 0.5

        rays_cam = pixel_to_camera_ray(u_f, v_f, cam)
        rays_world = rays_cam @ r_cw.T

        dx = rays_world[:, 0]
        dy = rays_world[:, 1]
        dz = rays_world[:, 2]

        active = dz < -1e-6
        if not np.any(active):
            continue

        dx_a = dx[active]
        dy_a = dy[active]
        dz_a = dz[active]
        idx_a = idx[active]

        tx = np.full(dx_a.shape, np.inf, dtype=np.float64)
        ty = np.full(dy_a.shape, np.inf, dtype=np.float64)

        m = dx_a > 1e-9
        tx[m] = rel_maxx / dx_a[m]
        m = dx_a < -1e-9
        tx[m] = rel_minx / dx_a[m]

        m = dy_a > 1e-9
        ty[m] = rel_maxy / dy_a[m]
        m = dy_a < -1e-9
        ty[m] = rel_miny / dy_a[m]

        t_exit = np.minimum(tx, ty)
        valid_exit = np.isfinite(t_exit) & (t_exit > 0)
        if not np.any(valid_exit):
            continue

        dx_a = dx_a[valid_exit]
        dy_a = dy_a[valid_exit]
        dz_a = dz_a[valid_exit]
        idx_a = idx_a[valid_exit]
        t_exit = t_exit[valid_exit]

        hit = np.zeros(dx_a.shape, dtype=bool)
        t_lo = np.zeros(dx_a.shape, dtype=np.float64)
        t_hi = np.zeros(dx_a.shape, dtype=np.float64)
        t_prev = np.zeros(dx_a.shape, dtype=np.float64)

        for step in range(1, march_steps + 1):
            t = t_exit * (step / march_steps)

            xs = center_x + dx_a * t
            ys = center_y + dy_a * t
            zs = cam_z + dz_a * t

            dem_z = bilinear_sample_scalar(dem, dem_transform, xs, ys, fill_value=np.nan)
            f = zs - dem_z

            crossed = (~hit) & np.isfinite(f) & (f <= 0.0)
            t_lo[crossed] = t_prev[crossed]
            t_hi[crossed] = t[crossed]
            hit |= crossed

            still_open = (~hit) & np.isfinite(f)
            t_prev[still_open] = t[still_open]

        if not np.any(hit):
            continue

        dx_h = dx_a[hit]
        dy_h = dy_a[hit]
        dz_h = dz_a[hit]
        idx_h = idx_a[hit]
        lo = t_lo[hit].copy()
        hi = t_hi[hit].copy()

        for _ in range(bisection_steps):
            mid = 0.5 * (lo + hi)
            xs = center_x + dx_h * mid
            ys = center_y + dy_h * mid
            zs = cam_z + dz_h * mid

            dem_z = bilinear_sample_scalar(dem, dem_transform, xs, ys, fill_value=np.nan)
            f = zs - dem_z

            above = np.isfinite(f) & (f > 0.0)
            lo = np.where(above, mid, lo)
            hi = np.where(above, hi, mid)

        t_hit = 0.5 * (lo + hi)
        x_hit = center_x + dx_h * t_hit
        y_hit = center_y + dy_h * t_hit

        colors = bilinear_sample_rgb(ortho_rgb, ortho_transform, x_hit, y_hit)
        flat_out[idx_h] = colors

    return out


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    if args.alt <= 0:
        raise ValueError("La altitud debe ser > 0")
    if not (0 < args.hfov < 179):
        raise ValueError("hfov debe estar entre 0 y 179 grados")
    if args.src_width < 2 or args.src_height < 2:
        raise ValueError("src-width y src-height deben ser >= 2")

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

    bbox_3857 = estimate_fetch_bbox(
        center_x=center_x,
        center_y=center_y,
        cam=cam,
        pose=pose,
        margin=args.margin,
        terrain_extra_m=args.terrain_extra_m,
    )

    print(f"BBOX solicitado: {bbox_3857}")

    dem, dem_transform = fetch_wcs_dem(
        wcs_url=args.dem_wcs_url,
        coverage=args.dem_coverage,
        bbox_3857=bbox_3857,
        width_px=args.src_width,
        height_px=args.src_height,
        fmt=args.dem_format,
        version=args.dem_wcs_version,
        timeout_s=args.timeout,
    )

    center_ground_z = bilinear_sample_scalar(
        dem,
        dem_transform,
        np.array([center_x], dtype=np.float64),
        np.array([center_y], dtype=np.float64),
        fill_value=np.nan,
    )[0]

    if not np.isfinite(center_ground_z):
        raise RuntimeError("No se pudo muestrear la elevación del DEM en la posición de la cámara")

    ortho_rgb, ortho_transform = fetch_wms_image(
        wms_url=args.ortho_wms_url,
        layer=args.ortho_layer,
        bbox_3857=bbox_3857,
        width_px=args.src_width,
        height_px=args.src_height,
        image_format=args.ortho_format,
        version=args.ortho_wms_version,
        timeout_s=args.timeout,
    )

    out = render_virtual_view(
        cam=cam,
        pose=pose,
        center_x=center_x,
        center_y=center_y,
        bbox_3857=bbox_3857,
        dem=dem,
        dem_transform=dem_transform,
        ortho_rgb=ortho_rgb,
        ortho_transform=ortho_transform,
        center_ground_z=center_ground_z,
        march_steps=args.march_steps,
        bisection_steps=args.bisection_steps,
        chunk_size=args.chunk_size,
    )

    Image.fromarray(out).save(args.out)

    print(f"Imagen guardada en: {args.out}")
    print(f"Elevación terreno en cámara: {center_ground_z:.3f} m")
    print(f"Altitud cámara absoluta aprox: {center_ground_z + pose.alt_agl_m:.3f} m")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
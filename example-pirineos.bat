@echo off
setlocal

REM ============================================
REM Prueba con relieve evidente: Pirineos
REM - Vista oblicua para que el DEM se note más
REM - Orto PNOA por WMS
REM - MDT IGN por WCS 2.0.1
REM ============================================

set SCRIPT=terrain_cam.py

python "%SCRIPT%" ^
  --ortho-wms-url "https://www.ign.es/wms-inspire/pnoa-ma" ^
  --ortho-layer "OI.OrthoimageCoverage" ^
  --dem-wcs-url "https://servicios.idee.es/wcs-inspire/mdt" ^
  --dem-coverage "Elevacion4258_25" ^
  --dem-wcs-version "2.0.1" ^
  --dem-format "image/tiff" ^
  --lat 42.6330 ^
  --lon 0.0000 ^
  --alt 800 ^
  --roll 0 ^
  --pitch 20 ^
  --yaw 110 ^
  --hfov 70 ^
  --width 1280 ^
  --height 720 ^
  --src-width 1536 ^
  --src-height 1536 ^
  --timeout 40 ^
  --out "test_dem_pirineos.png"

echo.
if errorlevel 1 (
  echo ERROR al ejecutar la prueba.
  echo.
  echo Sugerencias:
  echo   1. Probar otro CoverageId del IGN
  echo   2. Subir src-width/src-height a 2048
  echo   3. Bajar pitch a 10 si la vista sale rara
) else (
  echo OK. Imagen generada: test_dem_pirineos.png
)

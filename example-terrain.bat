@echo off
setlocal

REM ============================================
REM Test sencillo para terrain_cam_wcs.py
REM - Vista nadir
REM - Madrid centro
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
  --lat 40.4168 ^
  --lon -3.7038 ^
  --alt 120 ^
  --roll 0 ^
  --pitch 0 ^
  --yaw 0 ^
  --hfov 70 ^
  --width 1280 ^
  --height 720 ^
  --src-width 1024 ^
  --src-height 1024 ^
  --timeout 30 ^
  --out "test_dem_madrid.png"

echo.
if errorlevel 1 (
  echo ERROR al ejecutar la prueba.
  echo.
  echo Si falla el WCS del IGN, lo siguiente a revisar es:
  echo   1. El CoverageId
  echo   2. El CRS nativo que espera esa cobertura
  echo   3. El formato devuelto por GetCoverage
) else (
  echo OK. Imagen generada: test_dem_madrid.png
)

echo.
pause
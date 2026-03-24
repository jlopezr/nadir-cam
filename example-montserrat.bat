@echo off
setlocal

REM ============================================
REM Prueba con relieve: Montserrat
REM - Zona del Monestir de Montserrat
REM - Vista oblicua suave para apreciar el relieve
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
  --lat 41.593333 ^
  --lon 1.837556 ^
  --alt 500 ^
  --roll 0 ^
  --pitch 12 ^
  --yaw 135 ^
  --hfov 70 ^
  --width 1280 ^
  --height 720 ^
  --src-width 1536 ^
  --src-height 1536 ^
  --timeout 40 ^
  --out "test_dem_montserrat.png"

echo.
if errorlevel 1 (
  echo ERROR al ejecutar la prueba.
  echo.
  echo Prueba estos cambios:
  echo   1. --pitch 8
  echo   2. --alt 350
  echo   3. --src-width 2048 --src-height 2048
) else (
  echo OK. Imagen generada: test_dem_montserrat.png
)

python nadir_cam.py ^
  --wms-url "https://www.ign.es/wms-inspire/pnoa-ma" ^
  --layer "OI.OrthoimageCoverage" ^
  --lat 41.593333 ^
  --lon 1.837556 ^
  --alt 500 ^
  --roll 0 ^
  --pitch 12 ^
  --yaw 135 ^
  --hfov 70 ^
  --width 1280 ^
  --height 720 ^
  --out "test_dem_montserrat-2.png"
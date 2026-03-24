# Nadir / Oblique Virtual Camera from WMS

`nadir_cam_rpy.py` genera una imagen de cámara virtual a partir de una ortofoto servida por WMS y una pose completa de cámara:

* latitud, longitud y altitud
* roll, pitch y yaw
* resolución de imagen y FOV horizontal

La idea es sencilla: se proyectan los rayos de la cámara sobre un suelo plano, se pide al WMS la ortofoto que cubre esa huella y luego se rectifica con una homografía para obtener la vista que “vería” la cámara.

## Qué hace

El script:

1. Convierte `lat/lon` a `EPSG:3857`.
2. Sitúa la cámara en esa posición con la altitud indicada.
3. Construye una cámara pinhole a partir de `width`, `height` y `hfov`.
4. Calcula los rayos de las esquinas de imagen.
5. Aplica la orientación de cámara (`roll`, `pitch`, `yaw`).
6. Intersecta esos rayos con el plano del suelo `z = 0`.
7. Solicita al WMS una imagen que cubra ese `BBOX`.
8. Aplica una transformación de perspectiva para obtener la imagen final.

## Suposiciones del modelo

Este mini proyecto usa varias simplificaciones intencionadas:

* terreno plano: `z_ground = 0`
* la altitud de entrada es **AGL** en metros
* la cámara y el dron están en el mismo punto
* no se modela relieve ni DEM
* no se modela distorsión de lente
* el mundo local se trata como **ENU** (`x=este`, `y=norte`, `z=arriba`)
* la ortofoto se solicita en `EPSG:3857`

Estas simplificaciones son razonables para pruebas rápidas, simulación básica o validación visual.

## Convención de orientación

La orientación de la cámara se interpreta con convención **aeronáutica clásica**:

* **roll**: giro alrededor del eje longitudinal
* **pitch**: giro alrededor del eje lateral
* **yaw**: giro alrededor del eje vertical

En la pose neutra:

* la cámara mira hacia abajo
* la parte superior de la imagen apunta al **norte**
* la derecha de la imagen apunta al **este**
* `yaw = 0` corresponde a **norte**
* `yaw` positivo gira en sentido **horario**

## Requisitos

Python 3.10+ recomendado.

Dependencias:

```bash
pip install numpy opencv-python pillow requests pyproj
```

## Uso

Ejemplo:

```bash
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
```

## Parámetros principales

* `--wms-url`: URL base del servicio WMS
* `--layer`: capa WMS a solicitar
* `--lat`: latitud de la cámara en grados
* `--lon`: longitud de la cámara en grados
* `--alt`: altitud AGL en metros
* `--roll`: roll en grados
* `--pitch`: pitch en grados
* `--yaw`: yaw en grados (`0=norte`, positivo horario)
* `--hfov`: campo de visión horizontal en grados
* `--width`: ancho de imagen en píxeles
* `--height`: alto de imagen en píxeles
* `--out`: fichero de salida
* `--wms-version`: versión del WMS, por defecto `1.3.0`
* `--format`: formato de imagen WMS, por defecto `image/png`
* `--timeout`: timeout de la petición HTTP
* `--margin`: margen extra aplicado al `BBOX`

## Salida

El script:

* guarda la imagen sintetizada en el fichero indicado por `--out`
* imprime por consola:

  * la ruta de salida
  * las esquinas proyectadas sobre el suelo en `EPSG:3857`
  * el `BBOX` solicitado al WMS

## Limitaciones

* el terreno es completamente plano
* no hay oclusiones por edificios
* no hay horizonte ni cielo
* no hay relieve ni modelo de elevaciones
* `EPSG:3857` no es una proyección local exacta para fotogrametría precisa
* en ángulos muy oblicuos algunos rayos pueden no intersectar correctamente el suelo
* la calidad final depende de la resolución y cobertura del WMS
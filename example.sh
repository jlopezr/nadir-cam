#!/bin/bash
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
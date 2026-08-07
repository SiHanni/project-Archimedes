#!/usr/bin/env bash
# 깊이 추정 ONNX 가중치를 ./models 로 받는다 (Docker 볼륨으로 워커에 주입).
#
# Depth Anything V2 small (Apache-2.0) — 시차(disparity)를 내는 모델이므로
# ARCHIMEDES_DEPTH_OUTPUT_KIND=inverse_affine 로 써야 한다.
# 스케일 융합이 **역깊이 공간에서** 아핀을 맞춘다.
#
# ⚠️ 파일명을 바꾸지 말 것 — .onnx 가 외부 가중치(.onnx_data)를 이름으로 참조한다.
set -euo pipefail
cd "$(dirname "$0")/.."

REPO="${ARCHIMEDES_DEPTH_HF_REPO:-onnx-community/depth-anything-v2-small-ONNX}"
VARIANT="${ARCHIMEDES_DEPTH_VARIANT:-model_quantized}"
DEST="${ARCHIMEDES_MODELS_DIR:-./models}"

mkdir -p "$DEST"
for f in "${VARIANT}.onnx" "${VARIANT}.onnx_data"; do
  url="https://huggingface.co/${REPO}/resolve/main/onnx/${f}"
  if [ -s "${DEST}/${f}" ]; then
    echo "==> 이미 있음: ${DEST}/${f}"
    continue
  fi
  echo "==> 받는 중: ${f}"
  curl -fSL --progress-bar -o "${DEST}/${f}" "$url"
done

cat <<MSG

==> 완료. .env 에 아래를 넣고 워커를 재기동하세요.

  ARCHIMEDES_DEPTH_BACKEND=onnx
  ARCHIMEDES_DEPTH_MODEL_FILE=${VARIANT}.onnx
  ARCHIMEDES_DEPTH_OUTPUT_KIND=inverse_affine

  docker compose up -d --build worker worker-consumer

계약 확인:  docker compose exec worker python scripts/check_models.py
MSG

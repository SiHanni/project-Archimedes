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


# ── SlimSAM (프롬프트 기반 분할, 40MB) ─────────────────────────────────
# 외형 폴백의 임계값 흔들림을 줄인다. 실측: 같은 반지 두 사진의 마스크 편차가
# 1.87배 → 1.04배. 깊이 경로가 성공하면 타지 않으므로 상시 비용은 없다.
SAM_DIR="${MODEL_DIR}/sam"
mkdir -p "$SAM_DIR"
for f in onnx/vision_encoder.onnx onnx/prompt_encoder_mask_decoder.onnx config.json preprocessor_config.json; do
  echo "==> SlimSAM $(basename "$f")"
  curl -sSL --fail -o "${SAM_DIR}/$(basename "$f")" \
    "https://huggingface.co/Xenova/slimsam-77-uniform/resolve/main/${f}"
done


# ── BiRefNet (학습 기반 누끼, fp16 467MB) ──────────────────────────────
# 외곽선 탭의 **본선**. 색 임계값(Otsu·채도) 경로로는 체인 같은 가는 선, 상자·
# 저울 위 구도, 반지 구멍을 원리적으로 못 가른다 — 실사진 10장 중 8장이 깨졌다.
# BiRefNet 은 10장 전부 물체만 잡고 구멍까지 파냈다. MIT 라이선스.
BIREF_DIR="${MODEL_DIR}/birefnet"
mkdir -p "$BIREF_DIR"
if [ ! -s "${BIREF_DIR}/model_fp16.onnx" ]; then
  echo "==> BiRefNet model_fp16.onnx (467MB)"
  curl -fSL --progress-bar -o "${BIREF_DIR}/model_fp16.onnx" \
    "https://huggingface.co/onnx-community/BiRefNet-ONNX/resolve/main/onnx/model_fp16.onnx"
fi

cat <<MSG

==> 완료. .env 에 아래를 넣고 워커를 재기동하세요.

  ARCHIMEDES_DEPTH_BACKEND=onnx
  ARCHIMEDES_DEPTH_MODEL_FILE=${VARIANT}.onnx
  ARCHIMEDES_DEPTH_OUTPUT_KIND=inverse_affine
  ARCHIMEDES_SAM_DIR=/models/sam
  ARCHIMEDES_MATTE_DIR=/models/birefnet

  docker compose up -d --build worker worker-consumer

계약 확인:  docker compose exec worker python scripts/check_models.py
MSG

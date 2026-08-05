# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml .
COPY app ./app
COPY scripts ./scripts
RUN pip install --upgrade pip && pip install . \
    # rapidocr_onnxruntime 이 **full opencv-python**(X11 의존)을 끌어와
    # headless 를 덮어쓴다 → 컨테이너에서 `libxcb.so.1` 없음으로 import 실패.
    # 우리는 GUI 가 필요 없으므로 headless 로 되돌린다.
    && pip uninstall -y opencv-python opencv-contrib-python || true \
    && pip install --force-reinstall --no-deps "opencv-python-headless>=4.10" \
    && python -c "import cv2, rapidocr_onnxruntime; print('cv2', cv2.__version__, 'ocr ok')"
# Run as root in dev image so /output volume is writable for DEBUG_SAVE_MASKS
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

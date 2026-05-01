# Face direction classifier — CPU, Python 3.11
# Build:  docker build -t face-direction-classifier .
# Run:    docker run --rm -v "$(pwd):/data:ro" face-direction-classifier /data/face-1.png
# (Görseli konteynıra mount edin; çıktı stdout'ta: left | right | straight)

FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    GLOG_minloglevel=2

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libatomic1 \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY face_direction_classifier.py classify_face_direction.py ./

# Çalışma anında ağ gerekmemesi için modeli imaja göm
ARG MODEL_URL=https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
RUN curl -fsSL "$MODEL_URL" -o face_landmarker.task

ENTRYPOINT ["python3", "classify_face_direction.py"]

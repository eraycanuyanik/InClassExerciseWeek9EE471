# Pose arm classifier — CPU, Python 3.11
# Build:  docker build -t pose-arm-classifier .
# Run:    docker run --rm -v "$(pwd):/data:ro" pose-arm-classifier /data/pose-1.jpg
# (Görseli konteynıra mount edin; çıktı stdout’ta: left | right | both | None)

FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    GLOG_minloglevel=2

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libatomic1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY arm_classifier.py classify_pose_arms.py ./

# Çalışma anında ağ gerekmemesi için modeli imaja al
ARG MODEL_URL=https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task
RUN curl -fsSL "$MODEL_URL" -o pose_landmarker_lite.task

ENTRYPOINT ["python3", "classify_pose_arms.py"]

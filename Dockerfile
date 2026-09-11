# Backend image for Hugging Face Spaces (Docker SDK).
#
# FastAPI + torch (CPU) + the openSMILE audio stack. This mirrors the Render
# build ordering: the Phase 1 pins (requirements.txt) install first, then the
# Phase 2B audio extras (requirements-audio.txt). Installing the audio extras
# keeps the already-installed Phase 1 pins -- they satisfy the audio packages'
# ranges -- so no scientific dependency the models were trained against is
# upgraded or downgraded.
FROM python:3.11-slim

# System library required by soundfile / audiofile to decode the dataset WAVs.
# (openSMILE ships its own SMILExtract binary in the wheel, so no extra apt.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps in two steps, matching render.yaml. requirements.txt
# carries the `--extra-index-url .../whl/cpu` line, so the CPU torch wheel is used.
COPY requirements.txt requirements-audio.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r requirements-audio.txt

# App code + committed model artifacts + reports (all tracked in git).
COPY . .

# Runtime config.
#  - ALLOWED_ORIGINS: "*" is safe here (credential-less public API). Tighten to
#    the Vercel origin later if you like, via the Space's Variables settings.
#  - MPLCONFIGDIR / XDG_CACHE_HOME: keep matplotlib + library caches in a
#    writable location inside the container.
ENV OMP_NUM_THREADS=1 \
    AUDIO_MAX_UPLOAD_MB=25 \
    ALLOWED_ORIGINS="*" \
    MPLCONFIGDIR=/tmp/matplotlib \
    XDG_CACHE_HOME=/tmp/.cache \
    PYTHONUNBUFFERED=1

# Hugging Face Spaces routes external traffic to this port (see README app_port).
EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]

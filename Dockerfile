FROM python:3.13-slim

WORKDIR /app

# Copy requirements.txt before copying source code.
# Docker builds layers in order and caches each one.  If a layer's inputs
# haven't changed, Docker reuses the cached result and skips that step.
# Putting requirements.txt first means the (slow) pip install layer is only
# invalidated when dependencies actually change — not on every code edit.
# Reversing the order would bust the cache on every commit and force a full
# reinstall (~80 packages, ~60 s) even when only a single .py file changed.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the artifacts the API needs at runtime.
# Raw data, training splits, ablation CSVs, and notebooks stay out of the image.
COPY src/     src/
COPY models/  models/
COPY config/  config/
COPY data/processed/test.csv data/processed/test.csv

# Make src.* importable from /app.
ENV PYTHONPATH=/app

EXPOSE 8000

# Production invocation: no --reload.
# --reload watches the filesystem for .py changes and restarts the worker
# on every edit — useful in development, wasteful in a container where the
# code is baked in and never changes while the container is running.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

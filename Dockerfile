# ---- Stage 1: build the frontend ----
FROM node:22-alpine AS frontend
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
# Empty base = same-origin /api URLs; the backend serves these files itself.
ENV VITE_API_BASE=""
RUN npm run build

# ---- Stage 2: backend + static frontend ----
FROM python:3.11-slim
WORKDIR /app
COPY backend/ ./
RUN pip install --no-cache-dir . --extra-index-url https://download.pytorch.org/whl/cpu
COPY --from=frontend /build/dist ./frontend-dist
# chmod: platforms like HF Spaces run the container as a non-root user
RUN mkdir -p /data && chmod 777 /data
ENV FRONTEND_DIST=/app/frontend-dist \
    STUDIO_DB=/data/studio.db
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

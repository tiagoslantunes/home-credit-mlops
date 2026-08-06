# Dockerfile for the Home Credit Default Risk serving API.
#
# Multi-stage build keeps the runtime image small: the build stage installs
# every Python wheel (including the heavy native ones — lightgbm, numpy,
# scipy, scikit-learn), the runtime stage copies only the resolved
# site-packages, the app, and the trained model artefact.

ARG PYTHON_VERSION=3.12

# --------------------------------------------------------------------------- #
# Build stage                                                                  #
# --------------------------------------------------------------------------- #
FROM python:${PYTHON_VERSION}-slim AS build

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

# System deps for compiling/linking the native wheels (libgomp1 = OpenMP runtime
# LightGBM needs at run-time too).
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install only the runtime subset of requirements — the full requirements.txt
# contains development and notebook tools we do not want shipped.
COPY requirements-serving.txt .
RUN pip install --upgrade pip \
    && pip install --prefix=/install -r requirements-serving.txt


# --------------------------------------------------------------------------- #
# Runtime stage                                                                #
# --------------------------------------------------------------------------- #
FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    HOME_CREDIT_PROJECT_ROOT=/app \
    PATH="/usr/local/bin:${PATH}"

# Runtime OpenMP only — gcc/g++ are build-time concerns.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the resolved Python environment.
COPY --from=build /install /usr/local

# Project package + serving layer + the artefacts the model needs at runtime.
# The .dockerignore strips data/01_raw, notebooks/, .git, .venv, etc. so the
# layer below is bounded to what serving actually loads.
COPY src/ src/
COPY app/ app/
COPY conf/base/parameters_data_cleaning.yml conf/base/parameters_data_cleaning.yml
COPY conf/base/parameters_data_feat_engineering.yml conf/base/parameters_data_feat_engineering.yml
COPY data/04_feature/cleaning_params.pkl data/04_feature/cleaning_params.pkl
COPY data/06_models/production_model.pkl data/06_models/production_model.pkl
COPY data/06_models/decision_threshold.json data/06_models/decision_threshold.json
COPY pyproject.toml .
COPY README.md .

# Make the local package importable (``home_credit_mlops``).
RUN pip install --no-deps -e .

# Non-root user for production safety.
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

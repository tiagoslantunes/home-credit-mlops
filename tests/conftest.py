"""Configuração partilhada de pytest para toda a test suite."""

from __future__ import annotations

import os

# Permite MLflow usar SQLite file store sem servidor — necessário nos testes
# locais onde não existe um MLflow Tracking Server ativo.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

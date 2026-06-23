"""
Logging Configuration
=====================
Structured logging via loguru with console + file sinks.
All project modules import `logger` from this module.

Usage
-----
    from src.utils.logger import logger

    logger.info("Model training started | n_samples={}", n)
    logger.success("XGBoost fit complete | AUC={:.4f}", auc)
    logger.warning("High PSI detected | PSI={:.4f}", psi)
    logger.error("Feature '{}' not found in dataset", col)
"""

import sys
from pathlib import Path

from loguru import logger

# ── Remove default handler ─────────────────────────────────────────────────
logger.remove()

# ── Console handler — INFO and above ──────────────────────────────────────
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
           "<level>{level: <8}</level> | "
           "<cyan>{name}</cyan>:<cyan>{line}</cyan> — "
           "<level>{message}</level>",
    level="INFO",
    colorize=True,
)

# ── File handler — DEBUG and above (rotating, 10 MB per file) ──────────────
_LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
_LOG_DIR.mkdir(exist_ok=True)

logger.add(
    _LOG_DIR / "hybrid_credit_scoring_{time:YYYY-MM-DD}.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} — {message}",
    level="DEBUG",
    rotation="10 MB",
    retention="30 days",
    compression="zip",
)

__all__ = ["logger"]

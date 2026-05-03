"""Shared utilities: seed setting, logging, path helpers."""
import logging
import random
from pathlib import Path
import numpy as np
import torch

_FMT = "%(asctime)s %(levelname)s %(name)s — %(message)s"


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_logger(name: str, log_file: str | None = None) -> logging.Logger:
    """Return a logger that writes to stdout and optionally to *log_file*."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter(_FMT))
        logger.addHandler(ch)
        logger.setLevel(logging.INFO)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # Only add file handler once
        if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(log_path.resolve())
                   for h in logger.handlers):
            fh = logging.FileHandler(log_path, encoding="utf-8")
            fh.setFormatter(logging.Formatter(_FMT))
            logger.addHandler(fh)
    return logger

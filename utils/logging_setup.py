"""Logging setup using Loguru."""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from utils.config_loader import get_project_root, load_config


def setup_logging() -> None:
    config = load_config()
    log_cfg = config.get("logging", {})
    level = log_cfg.get("level", "INFO")
    log_file = get_project_root() / log_cfg.get("file", "logs/cliniq.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level=level)
    logger.add(log_file, rotation="1 MB", retention=5, level=level)

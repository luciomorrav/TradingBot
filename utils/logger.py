import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(level: str = "INFO", log_dir: str = "logs"):
    """Configure structured logging to console + rotating file."""
    Path(log_dir).mkdir(exist_ok=True)

    root = logging.getLogger()
    _VALID_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    level_name = level.upper() if level.upper() in _VALID_LEVELS else "INFO"
    root.setLevel(getattr(logging, level_name))

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler (force UTF-8 on Windows to support emojis)
    if sys.platform == "win32":
        import io
        console = logging.StreamHandler(
            io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        )
    else:
        console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    root.addHandler(console)

    # Rotating file handler (max 100MB, keep 7 files)
    file_handler = RotatingFileHandler(
        Path(log_dir) / "bot.log",
        maxBytes=100 * 1024 * 1024,
        backupCount=7,
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    # Quiet noisy libraries
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("telegram").setLevel(logging.WARNING)
    logging.getLogger("aiosqlite").setLevel(logging.WARNING)
    logging.getLogger("rlp").setLevel(logging.WARNING)

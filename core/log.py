import sys
from pathlib import Path

from loguru import logger

from conf import app_config

log_format = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)

logger.remove()
if app_config.logging.console.enable:
    def write_console(message):
        """Write logs without crashing on characters unsupported by Windows' code page."""
        encoding = sys.stdout.encoding or "utf-8"
        safe_message = str(message).encode(encoding, errors="backslashreplace").decode(encoding)
        sys.stdout.write(safe_message)
        sys.stdout.flush()

    logger.add(sink=write_console, level=app_config.logging.console.level, format=log_format)
if app_config.logging.file.enable:
    path = Path(app_config.logging.file.path)
    path.mkdir(parents=True, exist_ok=True)
    logger.add(
        sink=path / "app.log",
        level=app_config.logging.file.level,
        format=log_format,
        rotation=app_config.logging.file.rotation,
        retention=app_config.logging.file.retention,
        encoding="utf-8"
    )

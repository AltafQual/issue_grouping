import logging

_APP_FORMAT = "%(asctime)s | %(filename)s:%(lineno)d | %(funcName)s | %(levelname)s | %(message)s"
_APP_DATE_FMT = "%Y-%m-%d %H:%M:%S"
_configured = False


def _setup_root_logger() -> None:
    global _configured
    if _configured:
        return
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_APP_FORMAT, datefmt=_APP_DATE_FMT))
        root.addHandler(handler)
    else:
        # Handlers exist (e.g. added by uvicorn/gunicorn) — update their format
        # so our messages still show file/function info.
        formatter = logging.Formatter(_APP_FORMAT, datefmt=_APP_DATE_FMT)
        for h in root.handlers:
            h.setFormatter(formatter)
    _configured = True


class AppLogger:
    """
    Central logger factory.

    Usage (at module level):
        from src.logger import AppLogger
        logger = AppLogger().get_logger(__name__)

    Configures the root Python logger once so that ALL loggers — whether
    created via AppLogger or via logging.getLogger(__name__) directly —
    emit the same format:
        YYYY-MM-DD HH:MM:SS | filename.py:lineno | func_name | LEVEL | message
    """

    @classmethod
    def get_logger(cls, name: str = __name__) -> logging.Logger:
        _setup_root_logger()
        return logging.getLogger(name)

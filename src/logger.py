import logging


class AppLogger:
    _logger = None

    @classmethod
    def get_logger(cls, name: str = __name__) -> logging.Logger:
        if cls._logger is None:
            logger = logging.getLogger(name)
            logger.setLevel(logging.INFO)
            if not logger.handlers:
                handler = logging.StreamHandler()
                fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                handler.setFormatter(logging.Formatter(fmt))
                logger.addHandler(handler)
            cls._logger = logger
        return cls._logger

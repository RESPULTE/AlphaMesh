import logging
import sys


def get_logger(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    """
    Returns a configured logger that prints the filename automatically.
    """
    logger = logging.getLogger(name)

    if logger.hasHandlers():
        # Avoid adding multiple handlers if logger is called multiple times
        return logger

    logger.setLevel(level)

    # Create console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)

    # Create formatter with filename, function name, line number, and message
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(filename)s:%(funcName)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    logger.propagate = False  # Prevent double logging if root logger exists

    return logger

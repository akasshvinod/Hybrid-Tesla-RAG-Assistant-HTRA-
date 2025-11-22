import logging # Handles info,warnings,errors in a structured way
from rich.logging import RichHandler

def get_logger(name: str):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler()]
        )
    return logger

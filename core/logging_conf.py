import logging
import sys

# Create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # You can use DEBUG for more details

# Prevent duplicate handlers if file is reloaded
if not logger.hasHandlers():
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

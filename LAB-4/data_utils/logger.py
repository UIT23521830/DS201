# data_utils/logger.py
import logging
import os
import sys

def get_logger(bai_number, to_file=True):
    os.makedirs("logs", exist_ok=True)
    logger = logging.getLogger(f"bai{bai_number}")
    logger.setLevel(logging.INFO)

    # clear previous handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if to_file:
        fh = logging.FileHandler(f"logs/bai{bai_number}.log", mode="w", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger

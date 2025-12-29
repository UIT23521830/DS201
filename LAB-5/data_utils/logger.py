import logging
import os
from datetime import datetime


def get_logger(exp_id=0, log_dir="logs"):
    """
    Tạo logger ghi ra cả console và file
    exp_id: dùng để phân biệt các bài / thí nghiệm
    """

    os.makedirs(log_dir, exist_ok=True)

    log_name = f"bai{exp_id}.log"
    log_path = os.path.join(log_dir, log_name)

    logger = logging.getLogger(f"Experiment-{exp_id}")
    logger.setLevel(logging.INFO)

    # tránh add handler nhiều lần khi chạy lại notebook
    if logger.handlers:
        return logger

    # format log
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # file handler
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.info("Logger initialized")
    logger.info(f"Log file: {log_path}")

    return logger

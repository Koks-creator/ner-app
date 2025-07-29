import sys
from pathlib import Path
import os
from logging import Logger
sys.path.append(str(Path(__file__).resolve().parent.parent))
from fastapi import FastAPI

from predictor import NerPredictor
from config import Config
from custom_logger import CustomLogger

def setup_logging() -> Logger:
    """Configure logging for the api"""
    log_dir = os.path.dirname(Config.API_LOG_FILE)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = CustomLogger(
        logger_name="middleware_logger",
        logger_log_level=Config.CLI_LOG_LEVEL,
        file_handler_log_level=Config.FILE_LOG_LEVEL,
        log_file_name=Config.API_LOG_FILE
    ).create_logger()

    return logger

logger = setup_logging()
logger.info("Starting api...")

app = FastAPI(title="NerApi")
ner_predictor = NerPredictor(
    model_path=fr"{Config.MODEL_PATH}/{Config.MODEL_NAME}",
    word2idx_path=fr"{Config.MODEL_PATH}/{Config.WORD2IDX}",
    idx2tag_path=fr"{Config.MODEL_PATH}/{Config.IDX2TAG}",
    max_len=Config.MAX_LEN
)

from api import routes
logger.info("Api started")
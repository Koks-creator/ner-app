from pathlib import Path
import json
import os
from typing import Union
import logging


class Config:
    ROOT_PATH: Path = Path(__file__).resolve().parent
    MODELS_FOLDER: Path = rf"{ROOT_PATH}/models"

    # Logging
    CLI_LOG_LEVEL: int = logging.INFO
    FILE_LOG_LEVEL: int = logging.INFO

    # Model params
    MODEL_PATH: Path = rf"{MODELS_FOLDER}/model1"
    MODEL_NAME: str = "ner_model_2025_07_13.h5"
    WORD2IDX: str = "word2idx_2025_07_13.pkl"
    IDX2TAG: str = "idx2tag_2025_07_13.pkl"
    MAX_LEN: int = 70
    HUMAN_READABLE_TAGS_MAP = {
        "geo": "geographic locations",
        "gpe": "geopolitical entity",
        "tim": "time",
        "org": "organization",
        "per": "person",
        "art": "artifact",
        "nat": "nationality",
        "eve": "event"
    }
    
    # API
    API_PORT: int = 5000
    API_HOST: str = "127.0.0.1"
    API_URL: str = f"http://{API_HOST}:{API_PORT}"
    API_LOG_FILE: str = f"{ROOT_PATH}/api/logs/api_logs.log"

    # WEB APP
    WEB_APP_PORT: int = 8000
    WEB_APP_HOST: str = "127.0.0.1"
    WEB_APP_DEBUG: bool = True
    WEB_APP_LOG_FILE: str = f"{ROOT_PATH}/web_app/logs/web_app.logs"
    WEB_APP_TESTING: bool = False
    WEB_APP_MAX_FILES: int = 20

    # LOGGER
    UVICORN_LOG_CONFIG_PATH: Union[str, os.PathLike, Path] = f"{ROOT_PATH}/api/uvicorn_log_config.json"
    CLI_LOG_LEVEL: int = logging.DEBUG
    FILE_LOG_LEVEL: int = logging.DEBUG

    def get_uvicorn_logger(self) -> dict:
        with open(self.UVICORN_LOG_CONFIG_PATH) as f:
            log_config = json.load(f)
            log_config["handlers"]["file_handler"]["filename"] = f"{Config.ROOT_PATH}/api/logs/api_logs.log"
            return log_config

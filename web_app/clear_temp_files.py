"""
Turbo cleaner of temp files, use it in some schedule/crontab or use redis like a 
normal human being to store responds from api
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import os
from time import time
from glob import glob

from config import Config
from custom_logger import CustomLogger

logger = CustomLogger(
        logger_name="webapp_clear_temp_files",
        logger_log_level=Config.CLI_LOG_LEVEL,
        file_handler_log_level=Config.FILE_LOG_LEVEL,
        log_file_name=f"{Config.ROOT_PATH}/web_app/logs/webapp_clear_temp_files.logs"
    ).create_logger()


all_files = glob(f"{Config.WEB_APP_TEMP_FOLDER}/*")
now_timestamp = time()
logger.info(f"Starting clearing temp files: {now_timestamp=}")
summary = {
    "Failed": 0,
    "Deleted": 0
}
for file_path in all_files:
    _, filename = os.path.split(file_path)
    _, _, file_timestamp = filename.split("_")
    file_timestamp = int(file_timestamp.replace(".json", ""))
    diff = now_timestamp - file_timestamp
    if diff >= Config.WEB_APP_SESSION_TIME:
        try:
            os.remove(file_path)
            logger.info(f"Temp file deleted: {file_path=}")
            summary["Deleted"] += 1
        except Exception as e:
            logger.error(f"Temp file not deleted: {file_path=}, {e=}", exc_info=True)
            summary["Failed"] += 1
logger.info(f"{summary=}")

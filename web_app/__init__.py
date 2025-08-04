import sys
import os
import logging
from pathlib import Path
from flask import Flask
from dotenv import load_dotenv
import requests

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import Config

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")


def setup_logging(app: Flask) -> None:
    """Configure logging for the application"""
    log_dir = os.path.dirname(Config.WEB_APP_LOG_FILE)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    file_handler = logging.FileHandler(Config.WEB_APP_LOG_FILE)
    file_handler.setLevel(Config.FILE_LOG_LEVEL)
    
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - Line: %(lineno)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    
    app.logger.addHandler(file_handler)
    app.logger.setLevel(Config.FILE_LOG_LEVEL)
    app.logger.propagate = False  # Prevent duplicate logs

def create_app() -> Flask:
    """Initialize and configure the Flask application"""
    app = Flask(
        "NerWebApp",
        template_folder=TEMPLATE_DIR,
        static_folder=STATIC_DIR
    )
    os.makedirs(Config.WEB_APP_TEMP_FOLDER, exist_ok=True)
    setup_logging(app)
    app.logger.info("Starting web app")
    try:
        req = requests.get(Config.API_URL)
        if req.status_code != 200:
            app.logger.error(f"Could not connect to the api: {req.status_code=}")
    except Exception:
         app.logger.error("Could not connect to the api", exc_info=True)
    
    app.secret_key = os.getenv("SECRET_KEY")
    app.config["TESTING"] = Config.WEB_APP_TESTING
    app.config['PERMANENT_SESSION_LIFETIME'] = Config.WEB_APP_SESSION_TIME
    
    app.logger.info("Starting predictor")
    
    return app


app = create_app()

from web_app import routes

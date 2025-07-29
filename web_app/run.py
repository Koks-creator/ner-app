import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from web_app import app
from config import Config

def run() -> None:
    app.run(host=Config.WEB_APP_HOST, port=Config.WEB_APP_PORT, debug=Config.WEB_APP_DEBUG)


if __name__ == "__main__":
    run()
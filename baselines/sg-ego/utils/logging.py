import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import warnings

logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

logging.getLogger("httpx").setLevel(logging.WARNING)

warnings.filterwarnings(
    "ignore",
    message=".*The fast path is not available.*"
)

BASE_DIR = Path(
    os.environ.get(
        "SG_EGO_LOG_DIR",
        Path(__file__).resolve().parents[1] / "logs",
    )
)


class StdoutColorFormatter(logging.Formatter):
    RESET = "\033[0m"
    TIME_COLOR = "\033[36m"
    LEVEL_COLORS = {
        "DEBUG": "\033[37m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[35m",
    }

    def format(self, record: logging.LogRecord) -> str:
        timestamp = f"{self.TIME_COLOR}{self.formatTime(record, self.datefmt)}{self.RESET}"
        level_color = self.LEVEL_COLORS.get(record.levelname, self.RESET)
        levelname = f"{level_color}{record.levelname}{self.RESET}"
        message = record.getMessage()
        formatted = f"{timestamp} {levelname} {message}"

        if record.exc_info:
            formatted = f"{formatted}\n{self.formatException(record.exc_info)}"

        if record.stack_info:
            formatted = f"{formatted}\n{self.formatStack(record.stack_info)}"

        return formatted


def configure_logging(name: str) -> None:
    LOG_DIR = BASE_DIR / name
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_FILE = LOG_DIR / f"{timestamp}.log"

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(StdoutColorFormatter(datefmt="%Y-%m-%d %H:%M:%S"))

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))

    logging.basicConfig(
        level=logging.INFO,
        handlers=[stream_handler, file_handler],
        force=True,
    )

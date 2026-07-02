import logging
import time
from pathlib import Path

import yaml


class BaseCollector:
    """Base class for all data collectors.

    Handles config loading and logger setup so subclasses only need to
    implement their fetch logic.
    """

    CONFIG_PATH = Path(__file__).parents[2] / "config" / "config.yaml"

    def __init__(self):
        self.config = self._load_config()
        self.logger = self._setup_logger()

    def _load_config(self) -> dict:
        """Load and return the YAML config as a plain dict."""
        with open(self.CONFIG_PATH, "r") as f:
            return yaml.safe_load(f)

    def _setup_logger(self) -> logging.Logger:
        """Configure a logger for this collector.

        Writes to stdout always; also writes to a rotating log file in
        logs/ when log_to_file is true in config.
        """
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(self.config["logging"]["level"])

        if logger.handlers:
            # Avoid adding duplicate handlers if class is instantiated more
            # than once in the same process.
            return logger

        formatter = logging.Formatter(
            "%(asctime)s  %(name)s  %(levelname)s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        if self.config["logging"].get("log_to_file"):
            log_dir = Path(self.config["paths"]["log_dir"])
            log_dir.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_dir / f"{self.__class__.__name__}.log")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger

    def fetch_with_retry(self, fetch_function, max_retries: int, delay_seconds: float):
        """Call fetch_function, retrying on failure.

        Waits delay_seconds between each attempt. After exhausting
        max_retries the original exception is re-raised so the caller
        can decide how to handle the final failure.

        Args:
            fetch_function: A zero-argument callable that performs the
                fetch and returns its result.
            max_retries: Maximum number of attempts before giving up.
            delay_seconds: Seconds to wait between attempts.

        Returns:
            Whatever fetch_function returns on success.

        Raises:
            Exception: The last exception raised by fetch_function after
                all retries are exhausted.
        """
        last_exc = None
        for attempt in range(1, max_retries + 1):
            try:
                return fetch_function()
            except Exception as exc:
                last_exc = exc
                self.logger.warning(
                    "Attempt %d/%d failed: %s", attempt, max_retries, exc
                )
                if attempt < max_retries:
                    time.sleep(delay_seconds)
        raise last_exc

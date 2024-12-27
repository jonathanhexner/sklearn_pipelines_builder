import logging
import sys
from sklearn_pipelines_builder.utils.logger import logger

# Create a custom stream to redirect stdout
class RedirectStdOut:
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, message):
        if message.strip():  # Ignore empty lines
            self.logger.log(self.log_level, message.strip())

    def flush(self):
        pass  # Not needed for logger

# Redirect stdout to the logger
sys.stdout = RedirectStdOut(logger, logging.INFO)


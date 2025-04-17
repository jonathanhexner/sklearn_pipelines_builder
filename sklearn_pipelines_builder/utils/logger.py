import logging
from datetime import datetime
now =  datetime.now()
now_str = now.strftime('%Y_%m_%d-%H_%M')

#
# def get_logger(log_name='NER', file_name='./run.log'):
#     # set up logging to file - see previous section for more details
#     logging.basicConfig(level=logging.INFO,
#                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
#                        datefmt='%m-%d %H:%M',
#                        filename=file_name,
#                        filemode='w')
#     # define a Handler which writes INFO messages or higher to the sys.stderr
#     logFormatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
#     console = logging.StreamHandler()
#     console.setFormatter(logFormatter)
#     console.setLevel(logging.INFO)
#     logger = logging.getLogger(log_name)
#
#     if len(logger.handlers) == 0:
#         logger.addHandler(console)
#     return logger


class LoggerSingleton:
    _instances = {}  # Store logger instances
    _file_handlers = {}  # Store file handlers separately

    @classmethod
    def get_logger(cls, log_name='NER'):
        """Retrieve the logger instance (or create if it doesn't exist)."""
        if log_name not in cls._instances:
            # Create logger if it doesn't exist
            logger = logging.getLogger(log_name)
            logger.setLevel(logging.INFO)

            # Console handler (prints to screen)
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s'))
            console_handler.setLevel(logging.INFO)

            if not logger.handlers:
                logger.addHandler(console_handler)

            cls._instances[log_name] = logger  # Store instance

        return cls._instances[log_name]

    @classmethod
    def update_log_file(cls, log_name='NER', new_file_path='./new_log.log'):
        """Updates the log file path dynamically without affecting the logger instance."""
        if log_name in cls._instances:
            logger = cls._instances[log_name]

            # Remove the old file handler if it exists
            if log_name in cls._file_handlers:
                logger.removeHandler(cls._file_handlers[log_name])

            # Create and add the new file handler
            file_handler = logging.FileHandler(new_file_path, mode='w')
            file_handler.setFormatter(logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s'))
            logger.addHandler(file_handler)

            # Store the new handler
            cls._file_handlers[log_name] = file_handler
        else:
            # If logger does not exist yet, create it and attach file handler
            logger = cls.get_logger(log_name)
            cls.update_log_file(log_name, new_file_path)

# Example usage
now = datetime.now().strftime('%Y_%m_%d-%H_%M')
logger = LoggerSingleton.get_logger('sklearn_pipelines_builder')


# logger = get_logger('sklearn_pipelines_builder', f'logger_{now_str}.txt')

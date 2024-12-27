import logging
from datetime import datetime
now =  datetime.now()
now_str = now.strftime('%Y_%m_%d-%H_%M')

def get_logger(log_name='NER', file_name='./run.log'):
    # set up logging to file - see previous section for more details
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                       datefmt='%m-%d %H:%M',
                       filename=file_name,
                       filemode='w')
    # define a Handler which writes INFO messages or higher to the sys.stderr
    logFormatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    console = logging.StreamHandler()
    console.setFormatter(logFormatter)
    console.setLevel(logging.INFO)
    logger = logging.getLogger(log_name)

    if len(logger.handlers) == 0:
        logger.addHandler(console)
    return logger

logger = get_logger('sklearn_pipelines_builder', f'logger_{now_str}.txt')

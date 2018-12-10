from logging.handlers import TimedRotatingFileHandler
import logging
import os


def get_logger(name='cie'):
    if name == '':
        name = 'cie'
    logs_prefix = "log.txt"
    logs_directory = "logs"
    if not os.path.exists(logs_directory):
        os.makedirs(logs_directory)
    log_handler = TimedRotatingFileHandler(os.path.join(logs_directory, logs_prefix),
                                           when="midnight", backupCount=30)
    log_formatter = logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(funcName)s:%(lineno)d %(message)s")
    log_handler.setFormatter(log_formatter)
    m_logger = logging.getLogger(name)
    m_logger.addHandler(log_handler)
    m_logger.setLevel(logging.DEBUG)
    return m_logger


def get_name(path):
    return path.split('/')[-1].split('.')[0]


# logger = get_logger(name=get_name(__file__))

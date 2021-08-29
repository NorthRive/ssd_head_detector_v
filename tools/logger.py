import logging
import os
from datetime import datetime

from config import get_config


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "%(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

config, unparsed = get_config()

current_log_dir = os.path.join(config.log_dir, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
if not os.path.exists(current_log_dir):
    os.mkdir(current_log_dir)
my_logger = get_logger(current_log_dir + '/exp.log')

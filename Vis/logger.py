import logging
import os
import sys

def setup_logger(name, save_dir=None, control_print=True, use_formatter =True,filename="log.txt"):
    """
        建立日志记录工具
    """
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
 
    if control_print:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.INFO)
        if use_formatter:
            ch.setFormatter(formatter)
        logger.addHandler(ch)

    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(logging.INFO)
        if use_formatter:
            fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

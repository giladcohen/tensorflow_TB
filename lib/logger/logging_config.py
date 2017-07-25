import os
import logging
import logging.config

def logging_config():
    dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    log_config_file = os.path.join(dir_path, 'logger', 'configs', 'dlt_py_logger.ini')
    if not os.path.isfile(log_config_file):
        err_str = 'Log configuration file not found:'.format(log_config_file)
        raise Exception(err_str)
    logging.config.fileConfig(log_config_file, disable_existing_loggers=False)
    return logging

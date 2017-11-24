import os
import logging
import logging.config
import ConfigParser


def logging_config(log_file):
    dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    log_config_file = os.path.join(dir_path, 'logger', 'configs', 'dlt_py_logger.ini')
    if not os.path.isfile(log_config_file):
        err_str = 'Log configuration file not found:'.format(log_config_file)
        raise Exception(err_str)
    config = ConfigParser.RawConfigParser()
    config.read(log_config_file)
    config.set('handler_consoleHandler', 'args', "('"+log_file+"',)")
    log_root = os.path.dirname(log_file)
    config_file = os.path.join(log_root, 'logger_config_file')
    with open(config_file, 'wb') as configfile:
        config.write(configfile)

    logging.config.fileConfig(config_file, disable_existing_loggers=False)
    return logging

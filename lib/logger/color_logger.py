import logging

#These are the sequences need to get colored ouput
RESET_SEQ = "\033[0m"
COLOR_SEQ = "\033[1;%dm"
BOLD_SEQ = "\033[1m"

def formatter_message(message, use_color = True):
    if use_color:
        message = message.replace("$RESET", RESET_SEQ).replace("$BOLD", BOLD_SEQ)
    else:
        message = message.replace("$RESET", "").replace("$BOLD", "")
    return message

COLORS = {
    'DEBUG': 34,        # BLUE color for DEBUG level    (10)
    'INFO': 32,         # GREEN color for INFO level    (20)
    'WARNING': 33,      # BROWN color for WARNING level (30)
    'ERROR': 31,        # RED color for ERROR level     (40)
    'CRITICAL': 30,     # BLACK color for ERROR level   (50)
}

class ColoredFormatter(logging.Formatter):
    def __init__(self, msg, use_color = True):
        logging.Formatter.__init__(self, msg)
        self.use_color = use_color

    def format(self, record):
        levelname = record.levelname
        if self.use_color and levelname in COLORS:
            levelname_color = COLOR_SEQ % (COLORS[levelname]) + levelname + RESET_SEQ
            record.levelname = levelname_color
        return logging.Formatter.format(self, record)

def factory(fmt, datefmt):
    color_msg = "%(asctime)s- $BOLD%(name)-20s$RESET [%(levelname)-18s]  %(message)s ($BOLD%(filename)s$RESET:%(lineno)d)"
    color_format = formatter_message(color_msg, True)
    return ColoredFormatter(color_format)


def get_logger(logging_name, log_level=logging.DEBUG, color_on_console=True):
    " - %(name)s - %(levelname)s - %(message)s"
    msg = "%(asctime)s - %(name)-20s [ %(levelname)-6s ]:  %(message)s (%(filename)s :%(lineno)d)"
    formatter = logging.Formatter(msg)

    root_logger = logging.getLogger(logging_name)
    root_logger.setLevel(log_level)

    hl = root_logger.handlers
    if len(hl)  == 0:
        # Add handlers only if list of handlers is empty
        # Add console handler. Level INFO or higher will be send to stderr
        color_msg = "%(asctime)s- $BOLD%(name)-20s$RESET [%(levelname)-18s]  %(message)s ($BOLD%(filename)s$RESET:%(lineno)d)"
        color_format = formatter_message(color_msg, True)
        color_formatter = ColoredFormatter(color_format)

        console_handler = logging.StreamHandler()

        if color_on_console:
            console_handler.setFormatter(color_formatter)
        else:
            console_handler.setFormatter(formatter)

        root_logger.addHandler(console_handler)

    return root_logger

def main():
    color_on_colsole = True
    log_level = logging.DEBUG
    #logger = get_logger('main_test',log_level,color_on_colsole)
    logger.debug('DEBUG')
    logger.info('INFO')
    logger.warning('WARNING')
    logger.error('ERROR')
    logger.critical('CRITICAL')

if __name__ == '__main__':
    main()
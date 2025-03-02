from logging import Logger, INFO, NOTSET, FileHandler, Formatter, StreamHandler, addLevelName, setLoggerClass


Formatter_format = Formatter.format

# ANSI escape sequences for colors
BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)

RESET_SEQ = "\033[0m"
COLOR_SEQ = "\033[1;%dm"
BOLD_SEQ = "\033[1m"


COLORS = {
    'GOAL': BLUE,
    'FINISH': YELLOW,
}


class ColoredFormatter(Formatter):
    def __init__(self, msg, datefmt=None):
        super().__init__(msg, datefmt)

    def format(self, record):
        levelname = record.levelname
        if levelname in COLORS:
            color_code = 30 + COLORS[levelname]
            message_color = COLOR_SEQ % color_code + record.getMessage() + RESET_SEQ
            record.msg = message_color
        return Formatter_format(self, record)

class ColoredHandler(StreamHandler):
    def __init__(self, filepath=None, stream=None):  # filepath: log saved path
        super().__init__(stream)
        self.file_handler = None
        if filepath is not None:
            self.file_handler = FileHandler(filepath)

        colored_formatter = ColoredFormatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        self.setFormatter(colored_formatter)
        if self.file_handler:
            standard_formatter = Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
            self.file_handler.setFormatter(standard_formatter)

    def emit(self, record):
        original_msg = record.msg

        super().emit(record)

        if self.file_handler:
            record.msg = original_msg
            self.file_handler.emit(record)

class AgentLogger(Logger):
    GOAL_LEVEL_NUM = 100
    MESSAGE_LEVEL_NUM = 101
    ACTION_LEVEL_NUM = 102
    ACTION_INPUT_LEVEL_NUM = 103
    OBSERVATION_LEVEL_NUM = 104
    FINISH_LEVEL_NUM = 105

    def __init__(self, name, level=NOTSET, filepath=None):
        super().__init__(name, level)
        self.addHandler(ColoredHandler(filepath))
        self.setLevel(INFO)

    def goal(self, msg, *args, **kwargs):
        if self.isEnabledFor(self.GOAL_LEVEL_NUM):
            self._log(self.GOAL_LEVEL_NUM, msg, args, **kwargs)

    def finish(self, msg, *args, **kwargs):
        if self.isEnabledFor(self.FINISH_LEVEL_NUM):
            self._log(self.FINISH_LEVEL_NUM, msg, args, **kwargs)

addLevelName(AgentLogger.GOAL_LEVEL_NUM, "GOAL")
addLevelName(AgentLogger.FINISH_LEVEL_NUM, "FINISH")

setLoggerClass(AgentLogger)

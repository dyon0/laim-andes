from    dataclasses import dataclass


@dataclass(frozen = True)
class AnsiCodes:
    reset       : str   = '\033[0m'
    bold        : str   = '\033[1m'
    dim         : str   = '\033[2m'
    italic      : str   = '\033[3m'
    underline   : str   = '\033[4m'
    blink       : str   = '\033[5m'
    reverse     : str   = '\033[7m'
    hidden      : str   = '\033[8m'

    fg_black    : str   = '\033[30m'
    fg_red      : str   = '\033[31m'
    fg_green    : str   = '\033[32m'
    fg_yellow   : str   = '\033[33m'
    fg_blue     : str   = '\033[34m'
    fg_magenta  : str   = '\033[35m'
    fg_cyan     : str   = '\033[36m'
    fg_white    : str   = '\033[37m'
    fg_default  : str   = '\033[39m'

    bg_black    : str   = '\033[40m'
    bg_red      : str   = '\033[41m'
    bg_green    : str   = '\033[42m'
    bg_yellow   : str   = '\033[43m'
    bg_blue     : str   = '\033[44m'
    bg_magenta  : str   = '\033[45m'
    bg_cyan     : str   = '\033[46m'
    bg_white    : str   = '\033[47m'
    bg_default  : str   = '\033[49m'

    @staticmethod
    def fg_true(r: int, g: int, b: int) -> str:
        return f'\033[38;2;{r};{g};{b}m'

    @staticmethod
    def bg_true(r: int, g: int, b: int) -> str:
        return f'\033[48;2;{r};{g};{b}m'


@dataclass(frozen = True)
class ColorScheme:
    success : str   = ''
    error   : str   = ''
    warning : str   = ''
    info    : str   = ''
    debug   : str   = ''
    header  : str   = ''
    prompt  : str   = ''
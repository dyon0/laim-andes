from    dataclasses             import dataclass

from    ars.tools.tui.tui_core  import AnsiCodes
from    ars.tools.tui.tui       import ColorSchemeSystem, ColorSchemeDataScience


@dataclass(frozen = True)
class ColorSchemeRetroWave(ColorSchemeSystem):
    header      : str   = AnsiCodes.bg_true( 40,  10,  30) + AnsiCodes.fg_true(255, 255, 255) + AnsiCodes.bold
    subheader   : str   = AnsiCodes.fg_true(  0, 230, 230) + AnsiCodes.bold + AnsiCodes.underline
    success     : str   = AnsiCodes.fg_true( 57, 255,  20) + AnsiCodes.bold
    error       : str   = AnsiCodes.fg_true(255,  60,  60) + AnsiCodes.bold
    warning     : str   = AnsiCodes.fg_true(255, 165,   0) + AnsiCodes.bold + AnsiCodes.reverse
    info        : str   = AnsiCodes.fg_true(  0, 255, 255)
    debug       : str   = AnsiCodes.fg_true(180, 180, 180) + AnsiCodes.italic + AnsiCodes.dim
    prompt      : str   = AnsiCodes.fg_true(  0, 255, 255) + AnsiCodes.bold
    progress    : str   = AnsiCodes.fg_true(  0, 230, 230) + AnsiCodes.bold


@dataclass(frozen = True)
class ColorSchemeDataScienceCold(ColorSchemeDataScience):
    header              : str   = AnsiCodes.bg_true( 10,  25,  47) + AnsiCodes.fg_true(255, 255, 255) + AnsiCodes.bold
    subheader           : str   = AnsiCodes.fg_true(  0, 180, 216) + AnsiCodes.bold + AnsiCodes.underline
    success             : str   = AnsiCodes.fg_true( 46, 204, 113)
    warning             : str   = AnsiCodes.fg_true(243, 156,  18) + AnsiCodes.bold + AnsiCodes.reverse
    error               : str   = AnsiCodes.fg_true(231,  76,  60) + AnsiCodes.bold
    info                : str   = AnsiCodes.fg_true(144, 224, 239)
    debug               : str   = AnsiCodes.fg_true(131, 101, 181) + AnsiCodes.italic
    progress            : str   = AnsiCodes.fg_true(  0, 180, 216) + AnsiCodes.bold
    table_border        : str   = AnsiCodes.fg_true(  0, 119, 182)
    table_header        : str   = AnsiCodes.fg_true(  0, 180, 216) + AnsiCodes.bold
    table_cell          : str   = ''
    table_highlight     : str   = AnsiCodes.bg_true(  0, 180, 216) + AnsiCodes.fg_true(0, 0, 0) + AnsiCodes.bold
    best_metric         : str   = AnsiCodes.bg_true(255, 215,   0) + AnsiCodes.fg_true(0, 0, 0) + AnsiCodes.bold + AnsiCodes.blink
    perf_metric         : str   = AnsiCodes.bg_true(255, 215,   0) + AnsiCodes.fg_true(0, 0, 0) + AnsiCodes.bold
    good_metric         : str   = AnsiCodes.fg_true( 46, 204, 113)
    bad_metric          : str   = AnsiCodes.fg_true(231,  76,  60) + AnsiCodes.reverse
    progress_bar        : str   = AnsiCodes.fg_true(  0, 119, 182)
    progress_desc       : str   = AnsiCodes.fg_true(255, 255, 255)
    progress_color_hex  : str   = '#00B4D8'


@dataclass(frozen = True)
class ColorSchemeDataScienceWarm(ColorSchemeDataScience):
    header              : str   = AnsiCodes.bg_true( 50,  40,  30) + AnsiCodes.fg_true(255, 248, 220) + AnsiCodes.bold
    subheader           : str   = AnsiCodes.fg_true(217, 140,  46) + AnsiCodes.bold + AnsiCodes.underline
    success             : str   = AnsiCodes.fg_true( 74, 124,  89)
    warning             : str   = AnsiCodes.fg_true(230, 126,  34) + AnsiCodes.bold + AnsiCodes.reverse
    error               : str   = AnsiCodes.fg_true(192,  57,  43) + AnsiCodes.bold
    info                : str   = AnsiCodes.fg_true(230, 194, 106)
    debug               : str   = AnsiCodes.fg_true(140, 120, 100) + AnsiCodes.italic + AnsiCodes.dim
    progress            : str   = AnsiCodes.fg_true(217, 140,  46) + AnsiCodes.bold
    table_border        : str   = AnsiCodes.fg_true(212, 172,  13)
    table_header        : str   = AnsiCodes.fg_true(217, 140,  46) + AnsiCodes.bold
    table_cell          : str   = ''
    table_highlight     : str   = AnsiCodes.bg_true(217, 140,  46) + AnsiCodes.fg_true(0, 0, 0) + AnsiCodes.bold
    best_metric         : str   = AnsiCodes.bg_true(255, 215,   0) + AnsiCodes.fg_true(0, 0, 0) + AnsiCodes.bold + AnsiCodes.blink
    perf_metric         : str   = AnsiCodes.bg_true(255, 215,   0) + AnsiCodes.fg_true(0, 0, 0) + AnsiCodes.bold
    good_metric         : str   = AnsiCodes.fg_true( 74, 124,  89)
    bad_metric          : str   = AnsiCodes.fg_true(192,  57,  43) + AnsiCodes.reverse
    progress_bar        : str   = AnsiCodes.fg_true(217, 140,  46)
    progress_desc       : str   = AnsiCodes.fg_true(255, 248, 220)
    progress_color_hex  : str   = '#D98C2E'


@dataclass(frozen = True)
class ColorSchemeDataScienceSakura(ColorSchemeDataScience):
    header              : str   = AnsiCodes.bg_true(255, 192, 203) + AnsiCodes.fg_true( 80,  20,  60) + AnsiCodes.bold
    subheader           : str   = AnsiCodes.fg_true(255, 105, 180) + AnsiCodes.bold + AnsiCodes.underline
    success             : str   = AnsiCodes.fg_true(135, 206, 235)
    warning             : str   = AnsiCodes.fg_true(255, 180, 100) + AnsiCodes.blink + AnsiCodes.reverse
    error               : str   = AnsiCodes.fg_true(220,  20,  60) + AnsiCodes.bold
    info                : str   = AnsiCodes.fg_true(255, 192, 203)
    debug               : str   = AnsiCodes.fg_true(200, 160, 160) + AnsiCodes.italic + AnsiCodes.dim
    progress            : str   = AnsiCodes.fg_true(255, 105, 180) + AnsiCodes.bold
    table_border        : str   = AnsiCodes.fg_true(135, 206, 235)
    table_header        : str   = AnsiCodes.fg_true(255, 105, 180) + AnsiCodes.bold
    table_cell          : str   = AnsiCodes.fg_true(144, 238, 144)
    table_highlight     : str   = AnsiCodes.bg_true(255, 192, 203) + AnsiCodes.fg_true(0, 0, 0) + AnsiCodes.bold
    best_metric         : str   = AnsiCodes.bg_true(255,   0,   0) + AnsiCodes.fg_true(255, 255, 255) + AnsiCodes.bold + AnsiCodes.blink
    perf_metric         : str   = AnsiCodes.bg_true(255,   0,   0) + AnsiCodes.fg_true(255, 255, 255) + AnsiCodes.bold
    good_metric         : str   = AnsiCodes.fg_true(135, 206, 235)
    bad_metric          : str   = AnsiCodes.fg_true(220,  20,  60) + AnsiCodes.reverse
    progress_bar        : str   = AnsiCodes.fg_true(255, 192, 203)
    progress_desc       : str   = AnsiCodes.fg_true(255, 255, 255)
    progress_color_hex  : str   = '#FFC0CB'
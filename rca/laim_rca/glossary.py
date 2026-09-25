"""Словарь EPI-признаков детектора laim: семейство, описание, единица.

Ключ — базовое имя признака (`base` в detector_rca; агрегаты вроде
`duration_rolling_max_w5` приходят уже разобранными на base/aggregation/window).
Неизвестный признак не ломает анализ: он описывается своим именем.
"""
from __future__ import annotations

import re

# семейства признаков -> что они говорят аналитику
FAMILIES = {
    'timing':      'длительность и задержки',
    'llm':         'вызовы LLM',
    'tools':       'инструменты',
    'errors':      'ошибки',
    'output_text': 'форма выходного текста',
    'input_text':  'форма входного текста (промпт)',
    'structure':   'структура выполнения',
    'other':       'прочие признаки',
}

# base -> (семейство, описание, единица натуральной шкалы)
# единицы: ns — наносекунды (выводятся в с/мс), count, chars, ratio, share
FEATURES: dict[str, tuple[str, str, str]] = {
    'total_duration':               ('timing',      'полная длительность шага',                                'ns'),
    'duration':                     ('timing',      'длительность шага',                                       'ns'),
    'duration_diff':                ('timing',      'изменение длительности относительно предыдущего шага',    'ns'),
    'delta_time':                   ('timing',      'интервал между началами соседних шагов',                  'ns'),
    'exec_gap':                     ('timing',      'пауза между концом предыдущего шага и началом текущего',  'ns'),
    'llm_duration':                 ('timing',      'длительность вызова LLM',                                 'ns'),
    'tool_duration':                ('timing',      'длительность вызова инструмента',                         'ns'),
    'time_gap_variance':            ('timing',      'разброс интервалов между шагами',                         ''),
    'iteration_duration_variance':  ('timing',      'разброс длительности шагов',                              ''),
    'cv_duration':                  ('timing',      'коэффициент вариации длительности шагов',                 'ratio'),
    'cv_delta':                     ('timing',      'коэффициент вариации интервалов между шагами',            'ratio'),
    'llm_tokens':                   ('llm',         'число токенов вызова LLM',                                'count'),
    'token_ratio':                  ('llm',         'отношение длины входа к длине выхода LLM',                'ratio'),
    'processing_time_variance':     ('llm',         'разброс скорости генерации LLM',                          ''),
    'tool_success':                 ('tools',       'успешные вызовы инструментов',                            'count'),
    'unique_tools_local':           ('tools',       'число разных инструментов к этому шагу',                  'count'),
    'unique_tools_global':          ('tools',       'число разных инструментов в трассе',                      'count'),
    'tool_compression':             ('tools',       'отношение длины выхода инструмента к длине входа',        'ratio'),
    'error_flag':                   ('errors',      'маркер ошибки в тексте ответа',                           'count'),
    'char_count':                   ('output_text', 'длина ответа шага, символов',                             'chars'),
    'word_count':                   ('output_text', 'число слов в ответе шага',                                'count'),
    'sentence_count':               ('output_text', 'число предложений в ответе шага',                         'count'),
    'punctuation_count':            ('output_text', 'знаки препинания в ответе шага',                          'count'),
    'uppercase_count':              ('output_text', 'заглавные буквы в ответе шага',                           'count'),
    'digit_count':                  ('output_text', 'цифры в ответе шага',                                     'count'),
    'special_char_count':           ('output_text', 'спецсимволы в ответе шага',                               'count'),
    'avg_word_length_sem':          ('output_text', 'средняя длина слова в ответе шага',                       'chars'),
    'exec_out_char_count':          ('output_text', 'длина выхода шага, символов',                             'chars'),
    'exec_out_word_count':          ('output_text', 'число слов в выходе шага',                                'count'),
    'exec_out_sentence_count':      ('output_text', 'число предложений в выходе шага',                         'count'),
    'exec_out_punctuation_count':   ('output_text', 'знаки препинания в выходе шага',                          'count'),
    'exec_out_uppercase_count':     ('output_text', 'заглавные буквы в выходе шага',                           'count'),
    'exec_out_digit_count':         ('output_text', 'цифры в выходе шага',                                     'count'),
    'exec_out_special_char_count':  ('output_text', 'спецсимволы в выходе шага',                               'count'),
    'final_output_length':          ('output_text', 'длина финального ответа',                                 'chars'),
    'prompt_char_count':            ('input_text',  'длина входа шага, символов',                              'chars'),
    'prompt_word_count':            ('input_text',  'число слов во входе шага',                                'count'),
    'prompt_sentence_count':        ('input_text',  'число предложений во входе шага',                         'count'),
    'prompt_punctuation_count':     ('input_text',  'знаки препинания во входе шага',                          'count'),
    'prompt_uppercase_count':       ('input_text',  'заглавные буквы во входе шага',                           'count'),
    'prompt_digit_count':           ('input_text',  'цифры во входе шага',                                     'count'),
    'prompt_special_char_count':    ('input_text',  'спецсимволы во входе шага',                               'count'),
    'is_llm':                       ('structure',   'шаг — вызов LLM',                                         'share'),
    'is_tool':                      ('structure',   'шаг — вызов инструмента',                                 'share'),
    'is_chain':                     ('structure',   'шаг — цепочка (chain)',                                   'share'),
    'is_repeat':                    ('structure',   'повтор типа шага подряд',                                 'share'),
    'repetitive_actions':           ('structure',   'число повторов шагов в трассе',                           'count'),
    'unique_agents':                ('structure',   'число агентов в трассе',                                  'count'),
    'actions_per_agent':            ('structure',   'число шагов на агента',                                   'count'),
    'action_entropy':               ('structure',   'разнообразие типов шагов',                                ''),
}

_STATIC = {'max': 'максимум', 'min': 'минимум', 'mean': 'среднее', 'std': 'разброс', 'sum': 'сумма'}
_ROLLING = {'max': 'скользящий максимум', 'min': 'скользящий минимум', 'mean': 'скользящее среднее',
            'std': 'скользящий разброс', 'sum': 'скользящая сумма'}


def describe(base: str) -> tuple[str, str, str]:
    """(семейство, описание, единица) базового признака."""
    return FEATURES.get(base, ('other', base, ''))


def aggregation_phrase(aggregation: str | None, window: int | None) -> str:
    """'' для значения на шаге; иначе пояснение агрегата по последовательности агента."""
    if not aggregation:
        return ''
    rolling = aggregation.startswith('rolling_')
    name = aggregation.removeprefix('rolling_')
    if (quantile := re.fullmatch(r'q(\d+)', name)) is not None:
        label = f'{"скользящий " if rolling else ""}{quantile.group(1)}-й перцентиль'
    else:
        label = (_ROLLING if rolling else _STATIC).get(name, aggregation)
    return f'{label} по окну {window} шагов' if rolling and window else f'{label} по шагам агента'


def format_value(value: float, unit: str) -> str:
    """Значение в натуральных единицах для человека."""
    if unit == 'ns':
        seconds = value / 1e9
        if abs(seconds) >= 60:
            return f'{seconds / 60:.1f} мин'
        if abs(seconds) >= 1:
            return f'{seconds:.1f} с'
        return f'{seconds * 1000:.0f} мс'
    if unit in ('count', 'chars'):
        return f'{value:.0f}' if abs(value - round(value)) < 0.05 or abs(value) >= 100 else f'{value:.1f}'
    return f'{value:.2f}'

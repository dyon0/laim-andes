# ============================================================
# ИМПОРТЫ И ОБЩИЕ НАСТРОЙКИ
# ============================================================

from typing         import Literal, Tuple
from dataclasses    import dataclass
from functools      import partial
from itertools      import starmap, filterfalse
from pathlib        import PurePath
from json           import loads, dumps

import polars as pl

DATA_DIR : PurePath

parquet_path : PurePath

# ============================================================
# ОПРЕДЕЛЕНИЕ СПЕЦИФИКАЦИИ
# ============================================================

re_base64           = r'^[A-Za-z0-9+/]+={0,2}$'
re_code_base_elem   = r'^C[IE][0-9]+$'
re_json_str_array   = r'^\[\s*"([^"\\]|\\.)*"(\s*,\s*"([^"\\]|\\.)*")*\s*\]$'

status_code_values  = ('STATUS_CODE_UNSET', 'STATUS_CODE_OK', 'STATUS_CODE_ERROR')
aef_kind_values     = ('llm', 'start_agent', 'chain', 'tool', 'retriever', 'input_request', 'output_request', 'kafka_produce', 'kafka_consume', 'other', 'guard')
http_method_values  = ('GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS')


@dataclass(frozen = True)
class SpanDataAttr:
    group               : Literal[
        'Идентификаторы',
        'Время',
        'Статус',
        'Классификация',
        'Текстовые данные',
        'LLM',
        'HTTP',
        'Kafka',
        'Метаданные LangGraph',
        'Служебные',
        'Мета-атрибуты',
    ]
    name                : str
    parquet_type        : Literal[
        'BYTE_ARRAY (UTF8)',
        'INT64',
        'FLOAT',
        'BOOLEAN',
    ]
    polars_type         : type[pl.DataType] | pl.Enum
    constraints         : str
    sentinel            : None | Literal[
        '',
        -1,
        -1.0,
        False,
        'root',
        'outside',
        'STATUS_CODE_UNSET',
        'SPAN_KIND_UNSPECIFIED',
        'NONE',
    ]
    sentinel_semantic   : None | str
    semantic            : str
    computation         : str
    validation          : pl.Expr

    is_mandatory        : bool = False

    def __post_init__(self):
        object.__setattr__(self, 'validation', pl.col(self.name).is_not_null() & self.validation)


@dataclass(frozen = True)
class SpanDataSpec:
    attrs: Tuple[SpanDataAttr, ...]


def is_empty_or_valid_json(s: str, no_empty: bool = False) -> bool:
    if s == '': return not no_empty

    try:
        _ = loads(s)

        return True
    except Exception: return False


spec = SpanDataSpec(attrs = (
    SpanDataAttr(
        group               = 'Идентификаторы',
        name                = 'trace_id',
        parquet_type        = 'BYTE_ARRAY (UTF8)',
        polars_type         = pl.String,
        constraints         = 'NOT NULL, base64-строка',
        sentinel            = None,
        sentinel_semantic   = None,
        semantic            = 'Уникальный идентификатор трассы (одной сквозной цепочки вызовов) в рамках всего датасета.',
        computation         = 'traceId из JSON-спана',
        validation          = pl.col('trace_id').str.contains(re_base64),
        is_mandatory        = True,
    ),
    SpanDataAttr(
        group               = 'Идентификаторы',
        name                = 'span_id',
        parquet_type        = 'BYTE_ARRAY (UTF8)',
        polars_type         = pl.String,
        constraints         = 'NOT NULL, base64-строка, уникальна в паре с trace_id',
        sentinel            = None,
        sentinel_semantic   = None,
        semantic            = 'Уникальный идентификатор спана в рамках своей трассы.',
        computation         = 'spanId из JSON-спана',
        validation          = pl.col('span_id').str.contains(re_base64),
        is_mandatory        = True,
    ),
    SpanDataAttr(
        group               = 'Идентификаторы',
        name                = 'parent_span_id',
        parquet_type        = 'BYTE_ARRAY (UTF8)',
        polars_type         = pl.String,
        constraints         = 'NOT NULL, base64-строка или "root"',
        sentinel            = 'root',
        sentinel_semantic   = 'корень трассы (родительский спан отсутствует)',
        semantic            = 'Идентификатор родительского спана. "root" означает, что спан является корнем трассы.',
        computation         = 'parentSpanId; если отсутствует или равен span_id -> "root"',
        validation          = (pl.col('parent_span_id').str.contains(re_base64) | (pl.col('parent_span_id') == 'root')),
        is_mandatory        = True,
    ),
    SpanDataAttr(
        group               = 'Идентификаторы',
        name                = 'origin_span_id',
        parquet_type        = 'BYTE_ARRAY (UTF8)',
        polars_type         = pl.String,
        constraints         = 'NOT NULL, base64-строка или "outside"',
        sentinel            = 'outside',
        sentinel_semantic   = 'спан вне агента (нет предка start_agent)',
        semantic            = 'span_id корневого спана выполнения агента (ближайшего предка с aef_kind="start_agent"). "outside" — спан вне любого агента.',
        computation         = 'Рекурсивный подъём по parent_span_id до первого спана с aef_kind="start_agent"; если не найден -> "outside"',
        validation          = (pl.col('origin_span_id').str.contains(re_base64) | (pl.col('origin_span_id') == 'outside')),
        is_mandatory        = True,
    ),
    SpanDataAttr(
        group               = 'Идентификаторы',
        name                = 'agent_id',
        parquet_type        = 'BYTE_ARRAY (UTF8)',
        polars_type         = pl.String,
        constraints         = 'NOT NULL, КЭ модуля в формате "C[IE][0-9]+"',
        sentinel            = None,
        sentinel_semantic   = None,
        semantic            = 'КЭ (код элемента) модуля агента — идентификатор кодовой базы агента в реестре Сбера.',
        computation         = 'Из HEADER Kafka-сообщения, поле "agentid" (одинаково для всех спанов одного батча).',
        validation          = pl.col('agent_id').str.contains(re_code_base_elem),
        is_mandatory        = True,
    ),
    SpanDataAttr(
        group               = 'Идентификаторы',
        name                = 'session_id',
        parquet_type        = 'BYTE_ARRAY (UTF8)',
        polars_type         = pl.String,
        constraints         = 'NOT NULL, base64-строка',
        sentinel            = None,
        sentinel_semantic   = None,
        semantic            = 'Идентификатор пользовательской сессии.',
        computation         = 'aef.session_id в исходных данных. ВАЖНО! В случае, если в исходных данных у текущего спана атрибут сессии не установлен, тогда применяются правила заполнения session_id, описанные ниже.',
        validation          = pl.col('session_id').str.contains(re_base64),
        is_mandatory        = True,
    ),
    SpanDataAttr(
        group               = 'Время',
        name                = 'start_time_ns',
        parquet_type        = 'INT64',
        polars_type         = pl.Int64,
        constraints         = 'NOT NULL, >= 0',
        sentinel            = None,
        sentinel_semantic   = None,
        semantic            = 'Момент начала спана в наносекундах от Unix-эпохи (до 2262 г.).',
        computation         = 'startTimeUnixNano',
        validation          = pl.col('start_time_ns') >= 0,
        is_mandatory        = True,
    ),
    SpanDataAttr(
        group               = 'Время',
        name                = 'end_time_ns',
        parquet_type        = 'INT64',
        polars_type         = pl.Int64,
        constraints         = 'NOT NULL, >= start_time_ns',
        sentinel            = None,
        sentinel_semantic   = None,
        semantic            = 'Момент окончания спана (до 2262 г.).',
        computation         = 'endTimeUnixNano',
        validation          = pl.col('end_time_ns') >= pl.col('start_time_ns'),
        is_mandatory        = True,
    ),
    SpanDataAttr(
        group               = 'Статус',
        name                = 'status_code',
        parquet_type        = 'BYTE_ARRAY (UTF8)',
        polars_type         = pl.Enum(categories = status_code_values),
        constraints         = 'NOT NULL',
        sentinel            = 'STATUS_CODE_UNSET',
        sentinel_semantic   = 'статус не установлен (используется по умолчанию)',
        semantic            = 'Статус завершения спана: UNSET - не установлен, OK - успех, ERROR - ошибка.',
        computation         = 'status.code; если отсутствует -> STATUS_CODE_UNSET',
        validation          = pl.col('status_code').is_in(status_code_values),
    ),
    SpanDataAttr(
        group               = 'Статус',
        name                = 'status_message',
        parquet_type        = 'BYTE_ARRAY (UTF8)',
        polars_type         = pl.String,
        constraints         = 'NOT NULL, пустая строка "" если статус не ERROR',
        sentinel            = '',
        sentinel_semantic   = 'отсутствие ошибки или неприменимо',
        semantic            = 'Текст ошибки. Пустая строка "" означает отсутствие ошибки или неприменимость.',
        computation         = 'status.message; если отсутствует или статус не ERROR -> ""',
        validation          = (pl.when(pl.col('status_code') == 'STATUS_CODE_ERROR')
                                    .then(pl.col('status_message').str.len_chars() > 0)
                                    .otherwise(pl.col('status_message') == '')),
    ),
    SpanDataAttr(
        group               = 'Классификация',
        name                = 'aef_kind',
        parquet_type        = 'BYTE_ARRAY (UTF8)',
        polars_type         = pl.Enum(categories = aef_kind_values),
        constraints         = 'NOT NULL',
        sentinel            = None,
        sentinel_semantic   = None,
        semantic            = 'Семантический тип спана по классификации AEF.',
        computation         = 'Атрибут aef.kind',
        validation          = pl.col('aef_kind').is_in(aef_kind_values),
        is_mandatory        = True,
    ),
    SpanDataAttr(
        group               = 'Классификация',
        name                = 'span_name',
        parquet_type        = 'BYTE_ARRAY (UTF8)',
        polars_type         = pl.String,
        constraints         = 'NOT NULL',
        sentinel            = None,
        sentinel_semantic   = None,
        semantic            = 'Имя операции (название спана), например "GigaChat", "agent_start".',
        computation         = 'name из JSON-спана',
        validation          = pl.col('span_name').str.len_chars() > 0,
    ),
    SpanDataAttr(
        group               = 'Текстовые данные',
        name                = 'input_text',
        parquet_type        = 'BYTE_ARRAY (UTF8)',
        polars_type         = pl.String,
        constraints         = 'NOT NULL, пустая строка "" если вход отсутствует, иначе валидная JSON-строка (объект или массив, зависит от aef_kind)',
        sentinel            = '',
        sentinel_semantic   = 'вход отсутствует',
        semantic            = 'Входные данные спана. Для llm – JSON-массив сообщений, для tool/retriever – JSON-объект с параметрами, для input_request/output_request – тело HTTP-запроса (JSON-объект), для kafka_* – тело сообщения. "" – отсутствие входа.',
        computation         = 'Для llm,start_agent,chain,tool,retriever,guard,other – aef.input; для input_request,output_request – aef.request.body; для kafka_* – aef.kafka_body. Если атрибут отсутствует -> ""',
        validation          = pl.col('input_text').map_elements(is_empty_or_valid_json, return_dtype = pl.Boolean),
        is_mandatory        = True,
    ),
    SpanDataAttr(
        group               = 'Текстовые данные',
        name                = 'output_text',
        parquet_type        = 'BYTE_ARRAY (UTF8)',
        polars_type         = pl.String,
        constraints         = 'NOT NULL, пустая строка "" если выход отсутствует',
        sentinel            = '',
        sentinel_semantic   = 'выход отсутствует',
        semantic            = 'Выходные данные. Для llm - ответ модели, для tool - результат, для start_agent - итоговый ответ агента. "" - отсутствие выхода.',
        computation         = 'Для llm,start_agent,chain,tool,retriever,guard,other - aef.output; для input_request,output_request - aef.response.body. Если атрибут отсутствует -> ""',
        validation          = pl.lit(True),
        is_mandatory        = True,
    ),
    SpanDataAttr(
        group               = 'LLM',
        name                = 'llm_model',
        parquet_type        = 'BYTE_ARRAY (UTF8)',
        polars_type         = pl.String,
        constraints         = 'NOT NULL, "" если aef_kind != "llm"',
        sentinel            = '',
        sentinel_semantic   = 'неприменимо (спан не LLM)',
        semantic            = 'Название и версия вызванной LLM (например "GigaChat-2:2.0.28.2"). "" - неприменимо.',
        computation         = 'aef.llm.model; если отсутствует или не LLM -> ""',
        validation          = (pl.when(pl.col('aef_kind') == 'llm')
                                    .then(pl.col('llm_model').str.len_chars() > 0)
                                    .otherwise(pl.col('llm_model') == '')),
    ),
    SpanDataAttr(
        group               = 'LLM',
        name                = 'llm_prompt_tokens',
        parquet_type        = 'INT64',
        polars_type         = pl.Int64,
        constraints         = 'NOT NULL, -1 если aef_kind != "llm", иначе >= 0',
        sentinel            = -1,
        sentinel_semantic   = 'неприменимо (спан не LLM)',
        semantic            = 'Количество токенов во входящем сообщении (роль user). -1 - неприменимо.',
        computation         = 'aef.llm.prompt_tokens; если отсутствует или не LLM -> -1',
        validation          = (pl.when(pl.col('aef_kind') == 'llm')
                                    .then(pl.col('llm_prompt_tokens') >= 0)
                                    .otherwise(pl.col('llm_prompt_tokens') == -1)),
    ),
    SpanDataAttr(
        group               = 'LLM',
        name                = 'llm_completion_tokens',
        parquet_type        = 'INT64',
        polars_type         = pl.Int64,
        constraints         = 'NOT NULL, -1 если aef_kind != "llm", иначе >= 0',
        sentinel            = -1,
        sentinel_semantic   = 'неприменимо (спан не LLM)',
        semantic            = 'Количество токенов, сгенерированных моделью (роль assistant). -1 - неприменимо.',
        computation         = 'aef.llm.completion_tokens; если отсутствует или не LLM -> -1',
        validation          = (pl.when(pl.col('aef_kind') == 'llm')
                                    .then(pl.col('llm_completion_tokens') >= 0)
                                    .otherwise(pl.col('llm_completion_tokens') == -1)),
    ),
    SpanDataAttr(
        group               = 'LLM',
        name                = 'llm_total_tokens',
        parquet_type        = 'INT64',
        polars_type         = pl.Int64,
        constraints         = 'NOT NULL, -1 если aef_kind != "llm", иначе >= 0',
        sentinel            = -1,
        sentinel_semantic   = 'неприменимо (спан не LLM)',
        semantic            = 'Общее число тарифицируемых токенов. -1 - неприменимо.',
        computation         = 'aef.llm.total_tokens; если отсутствует или не LLM -> -1',
        validation          = (pl.when(pl.col('aef_kind') == 'llm')
                                    .then(pl.col('llm_total_tokens') >= 0)
                                    .otherwise(pl.col('llm_total_tokens') == -1)),
    ),
    SpanDataAttr(
        group               = 'LLM',
        name                = 'llm_precached_prompt_tokens',
        parquet_type        = 'INT64',
        polars_type         = pl.Int64,
        constraints         = 'NOT NULL, -1 если aef_kind != "llm", иначе >= 0',
        sentinel            = -1,
        sentinel_semantic   = 'неприменимо (спан не LLM)',
        semantic            = 'Количество кэшированных токенов. -1 - неприменимо.',
        computation         = 'aef.llm.precached_prompt_tokens; если отсутствует или не LLM -> -1',
        validation          = (pl.when(pl.col('aef_kind') == 'llm')
                                    .then(pl.col('llm_precached_prompt_tokens') >= 0)
                                    .otherwise(pl.col('llm_precached_prompt_tokens') == -1)),
    ),
    SpanDataAttr(
        group               = 'LLM',
        name                = 'llm_temperature',
        parquet_type        = 'FLOAT',
        polars_type         = pl.Float32,
        constraints         = 'NOT NULL, -1.0 если aef_kind != "llm", иначе >= 0.0',
        sentinel            = -1.0,
        sentinel_semantic   = 'неприменимо (спан не LLM)',
        semantic            = 'Температура (параметр вызова модели). -1.0 - неприменимо.',
        computation         = 'aef.llm.model_parameters.temperature; если отсутствует или не LLM -> -1.0',
        validation          = (pl.when(pl.col('aef_kind') == 'llm')
                                    .then(pl.col('llm_temperature') >= 0.0)
                                    .otherwise(pl.col('llm_temperature') == -1.0)),
    ),
    SpanDataAttr(
        group               = 'LLM',
        name                = 'llm_top_p',
        parquet_type        = 'FLOAT',
        polars_type         = pl.Float32,
        constraints         = 'NOT NULL, -1.0 если aef_kind != "llm", иначе 0.0..1.0',
        sentinel            = -1.0,
        sentinel_semantic   = 'неприменимо (спан не LLM)',
        semantic            = 'Top-p (параметр вызова модели). -1.0 - неприменимо.',
        computation         = 'aef.llm.model_parameters.top_p; если отсутствует или не LLM -> -1.0',
        validation          = (pl.when(pl.col('aef_kind') == 'llm')
                                    .then(pl.col('llm_top_p').is_between(0.0, 1.0))
                                    .otherwise(pl.col('llm_top_p') == -1.0)),
    ),
    SpanDataAttr(
        group               = 'LLM',
        name                = 'llm_max_tokens',
        parquet_type        = 'INT64',
        polars_type         = pl.Int64,
        constraints         = 'NOT NULL, -1 если aef_kind != "llm", иначе >= 0',
        sentinel            = -1,
        sentinel_semantic   = 'неприменимо (спан не LLM)',
        semantic            = 'Максимальное количество токенов в ответе (параметр). -1 - неприменимо.',
        computation         = 'aef.llm.model_parameters.max_tokens; если отсутствует или не LLM -> -1',
        validation          = (pl.when(pl.col('aef_kind') == 'llm')
                                    .then(pl.col('llm_max_tokens') >= 0)
                                    .otherwise(pl.col('llm_max_tokens') == -1)),
    ),
    SpanDataAttr(
        group               = 'LLM',
        name                = 'llm_repetition_penalty',
        parquet_type        = 'FLOAT',
        polars_type         = pl.Float32,
        constraints         = 'NOT NULL, -1.0 если aef_kind != "llm", иначе >= 1.0',
        sentinel            = -1.0,
        sentinel_semantic   = 'неприменимо (спан не LLM)',
        semantic            = 'Штраф за повторения. -1.0 - неприменимо.',
        computation         = 'aef.llm.model_parameters.repetition_penalty; если отсутствует или не LLM -> -1.0',
        validation          = (pl.when(pl.col('aef_kind') == 'llm')
                                    .then(pl.col('llm_repetition_penalty') >= 1.0)
                                    .otherwise(pl.col('llm_repetition_penalty') == -1.0)),
    ),
    SpanDataAttr(
        group               = 'LLM',
        name                = 'llm_profanity_check',
        parquet_type        = 'BOOLEAN',
        polars_type         = pl.Boolean,
        constraints         = 'NOT NULL, False если aef_kind != "llm"',
        sentinel            = False,
        sentinel_semantic   = 'неприменимо или отключено',
        semantic            = 'Включена ли проверка ненормативной лексики. False - неприменимо или отключено.',
        computation         = 'aef.llm.model_parameters.profanity_check; если отсутствует или не LLM -> False',
        validation          = (pl.when(pl.col('aef_kind') == 'llm')
                                    .then(pl.col('llm_profanity_check').is_in((True, False)))
                                    .otherwise(pl.col('llm_profanity_check') == False)),
    ),
    SpanDataAttr(
        group               = 'LLM',
        name                = 'llm_stream',
        parquet_type        = 'BOOLEAN',
        polars_type         = pl.Boolean,
        constraints         = 'NOT NULL, False если aef_kind != "llm"',
        sentinel            = False,
        sentinel_semantic   = 'неприменимо или отключено',
        semantic            = 'Потоковый режим. False - неприменимо или отключено.',
        computation         = 'aef.llm.model_parameters.stream; если отсутствует или не LLM -> False',
        validation          = (pl.when(pl.col('aef_kind') == 'llm')
                                    .then(pl.col('llm_stream').is_in((True, False)))
                                    .otherwise(pl.col('llm_stream') == False)),
    ),
    SpanDataAttr(
        group               = 'HTTP',
        name                = 'http_method',
        parquet_type        = 'BYTE_ARRAY (UTF8)',
        polars_type         = pl.Enum(categories = http_method_values + ('NONE',)),
        constraints         = 'NOT NULL, "NONE" если aef_kind not in {"input_request","output_request"}',
        sentinel            = 'NONE',
        sentinel_semantic   = 'неприменимо (спан не HTTP-запрос/ответ)',
        semantic            = 'HTTP-метод. "NONE" - неприменимо.',
        computation         = 'aef.request.method; если отсутствует или не HTTP -> "NONE"',
        validation          = (pl.when(pl.col('aef_kind').is_in(('input_request', 'output_request')))
                                    .then(pl.col('http_method').is_in(http_method_values))
                                    .otherwise(pl.col('http_method') == 'NONE')),
    ),
    SpanDataAttr(
        group               = 'HTTP',
        name                = 'http_path',
        parquet_type        = 'BYTE_ARRAY (UTF8)',
        polars_type         = pl.String,
        constraints         = 'NOT NULL, "" если aef_kind not in {"input_request","output_request"}',
        sentinel            = '',
        sentinel_semantic   = 'неприменимо (спан не HTTP-запрос/ответ)',
        semantic            = 'Путь HTTP-запроса (например "/chat"). "" - неприменимо.',
        computation         = 'aef.request.path; если отсутствует или не HTTP -> ""',
        validation          = (pl.when(pl.col('aef_kind').is_in(('input_request', 'output_request')))
                                    .then(pl.col('http_path').str.len_chars() > 0)
                                    .otherwise(pl.col('http_path') == '')),
    ),
    SpanDataAttr(
        group               = 'HTTP',
        name                = 'http_status_code',
        parquet_type        = 'INT64',
        polars_type         = pl.Int64,
        constraints         = 'NOT NULL, -1 если aef_kind not in {"input_request","output_request"}, иначе 100..599',
        sentinel            = -1,
        sentinel_semantic   = 'неприменимо (спан не HTTP-запрос/ответ)',
        semantic            = 'HTTP-статус ответа. -1 - неприменимо.',
        computation         = 'Парсится из aef.response.body или aef.response.headers; если не извлекается или не HTTP -> -1',
        validation          = (pl.when(pl.col('aef_kind').is_in(('input_request', 'output_request')))
                                    .then(pl.col('http_status_code').is_between(100, 599))
                                    .otherwise(pl.col('http_status_code') == -1)),
    ),
    SpanDataAttr(
        group               = 'HTTP',
        name                = 'request_headers',
        parquet_type        = 'BYTE_ARRAY (UTF8)',
        polars_type         = pl.String,
        constraints         = 'NOT NULL, "" если aef_kind not in {"input_request","output_request"}, иначе требуется валидная JSON-строка',
        sentinel            = '',
        sentinel_semantic   = 'неприменимо (спан не HTTP-запрос/ответ)',
        semantic            = 'Заголовки запроса (JSON-строка). "" - неприменимо.',
        computation         = 'aef.request.headers; если отсутствует или не HTTP -> ""',
        validation          = (pl.when(pl.col('aef_kind').is_in(('input_request', 'output_request')))
                                    .then(pl.col('request_headers').map_elements(partial(is_empty_or_valid_json, no_empty = True), return_dtype = pl.Boolean))
                                    .otherwise(pl.col('request_headers') == '')),
    ),
    SpanDataAttr(
        group               = 'HTTP',
        name                = 'response_headers',
        parquet_type        = 'BYTE_ARRAY (UTF8)',
        polars_type         = pl.String,
        constraints         = 'NOT NULL, "" если aef_kind not in {"input_request","output_request"}, иначе требуется валидная JSON-строка',
        sentinel            = '',
        sentinel_semantic   = 'неприменимо (спан не HTTP-запрос/ответ)',
        semantic            = 'Заголовки ответа (JSON-строка). "" - неприменимо.',
        computation         = 'aef.response.headers; если отсутствует или не HTTP -> ""',
        validation          = (pl.when(pl.col('aef_kind').is_in(('input_request', 'output_request')))
                                    .then(pl.col('response_headers').map_elements(partial(is_empty_or_valid_json, no_empty = True), return_dtype = pl.Boolean))
                                    .otherwise(pl.col('response_headers') == '')),
    ),

#    SpanDataAttr(
#        group               = 'HTTP',
#        name                = 'request_body',
#        parquet_type        = 'BYTE_ARRAY (UTF8)',
#        polars_type         = pl.String,
#        constraints         = 'NOT NULL, "" если aef_kind not in {"input_request","output_request"}, иначе требуется валидная JSON-строка',
#        sentinel            = '',
#        sentinel_semantic   = 'неприменимо (спан не HTTP-запрос/ответ)',
#        semantic            = 'Тело запроса (JSON-строка). "" - неприменимо.',
#        computation         = 'aef.request.body; если отсутствует или не HTTP -> ""',
#        validation          = (pl.when(pl.col('aef_kind').is_in(('input_request', 'output_request')))
#                                    .then(pl.col('request_body').map_elements(partial(is_empty_or_valid_json, no_empty = True), return_dtype = pl.Boolean))
#                                    .otherwise(pl.col('request_body') == '')),
#    ),
#    SpanDataAttr(
#        group               = 'HTTP',
#        name                = 'response_body',
#        parquet_type        = 'BYTE_ARRAY (UTF8)',
#        polars_type         = pl.String,
#        constraints         = 'NOT NULL, "" если aef_kind not in {"input_request","output_request"}, иначе требуется валидная JSON-строка',
#        sentinel            = '',
#        sentinel_semantic   = 'неприменимо (спан не HTTP-запрос/ответ)',
#        semantic            = 'Тело ответа (JSON-строка). "" - неприменимо.',
#        computation         = 'aef.response.body; если отсутствует или не HTTP -> ""',
#        validation          = (pl.when(pl.col('aef_kind').is_in(('input_request', 'output_request')))
#                                    .then(pl.col('response_body').map_elements(partial(is_empty_or_valid_json, no_empty = True), return_dtype = pl.Boolean))
#                                    .otherwise(pl.col('response_body') == '')),
#    ),

    SpanDataAttr(
        group               = 'Kafka',
        name                = 'kafka_topic',
        parquet_type        = 'BYTE_ARRAY (UTF8)',
        polars_type         = pl.String,
        constraints         = 'NOT NULL, "" если aef_kind not in {"kafka_produce","kafka_consume"}',
        sentinel            = '',
        sentinel_semantic   = 'неприменимо (спан не Kafka)',
        semantic            = 'Топик Kafka. "" - неприменимо.',
        computation         = 'aef.kafka_topic; если отсутствует или не Kafka -> ""',
        validation          = (pl.when(pl.col('aef_kind').is_in(('kafka_produce', 'kafka_consume')))
                                    .then(pl.col('kafka_topic').str.len_chars() > 0)
                                    .otherwise(pl.col('kafka_topic') == '')),
    ),
    SpanDataAttr(
        group               = 'Kafka',
        name                = 'kafka_cluster',
        parquet_type        = 'BYTE_ARRAY (UTF8)',
        polars_type         = pl.String,
        constraints         = 'NOT NULL, "" если aef_kind not in {"kafka_produce","kafka_consume"}',
        sentinel            = '',
        sentinel_semantic   = 'неприменимо (спан не Kafka)',
        semantic            = 'Идентификатор кластера Kafka. "" - неприменимо.',
        computation         = 'aef.kafka_cluster; если отсутствует или не Kafka -> ""',
        validation          = (pl.when(pl.col('aef_kind').is_in(('kafka_produce', 'kafka_consume')))
                                    .then(pl.col('kafka_cluster').str.len_chars() > 0)
                                    .otherwise(pl.col('kafka_cluster') == '')),
    ),
    SpanDataAttr(
        group               = 'Kafka',
        name                = 'kafka_consumer_group',
        parquet_type        = 'BYTE_ARRAY (UTF8)',
        polars_type         = pl.String,
        constraints         = 'NOT NULL, "" если aef_kind != "kafka_consume"',
        sentinel            = '',
        sentinel_semantic   = 'неприменимо (спан не Kafka consumer)',
        semantic            = 'Группа потребителей Kafka. "" - неприменимо.',
        computation         = 'aef.consumer_group; если отсутствует или не consumer -> ""',
        validation          = (pl.when(pl.col('aef_kind') == 'kafka_consume')
                                    .then(pl.col('kafka_consumer_group').str.len_chars() > 0)
                                    .otherwise(pl.col('kafka_consumer_group') == '')),
    ),
    SpanDataAttr(
        group               = 'Kafka',
        name                = 'kafka_bootstrap_servers',
        parquet_type        = 'BYTE_ARRAY (UTF8)',
        polars_type         = pl.String,
        constraints         = f'NOT NULL, "" если aef_kind not in {"kafka_produce","kafka_consume"}, иначе строка, соответствующая регулярному выражению: {re_json_str_array}',
        sentinel            = '',
        sentinel_semantic   = 'неприменимо (спан не Kafka)',
        semantic            = 'Список bootstrap-серверов (JSON-массив строк, например ["host1:9092","host2:9092"]). "" - неприменимо.',
        computation         = 'aef.bootstrap_servers; если отсутствует или не Kafka -> ""',
        validation          = (pl.when(pl.col('aef_kind').is_in(('kafka_produce', 'kafka_consume')))
                                    .then(pl.col('kafka_bootstrap_servers').str.contains(re_json_str_array))
                                    .otherwise(pl.col('kafka_bootstrap_servers') == '')),
    ),
    SpanDataAttr(
        group               = 'Метаданные LangGraph',
        name                = 'meta_langgraph_step',
        parquet_type        = 'INT64',
        polars_type         = pl.Int64,
        constraints         = 'NOT NULL, -1 если отсутствует',
        sentinel            = -1,
        sentinel_semantic   = 'отсутствует',
        semantic            = 'Номер шага в графе LangGraph. -1 - отсутствует.',
        computation         = 'aef.metadata.langgraph_step; иначе -1',
        validation          = pl.col('meta_langgraph_step') >= -1,
    ),
    SpanDataAttr(
        group               = 'Метаданные LangGraph',
        name                = 'meta_langgraph_node',
        parquet_type        = 'BYTE_ARRAY (UTF8)',
        polars_type         = pl.String,
        constraints         = 'NOT NULL, "" если отсутствует',
        sentinel            = '',
        sentinel_semantic   = 'отсутствует',
        semantic            = 'Имя текущего узла графа. "" - отсутствует.',
        computation         = 'aef.metadata.langgraph_node; иначе ""',
        validation          = pl.lit(True),
    ),
    SpanDataAttr(
        group               = 'Метаданные LangGraph',
        name                = 'meta_langgraph_triggers',
        parquet_type        = 'BYTE_ARRAY (UTF8)',
        polars_type         = pl.String,
        constraints         = f'NOT NULL, "" если отсутствует, иначе строка, соответствующая регулярному выражению: {re_json_str_array}',
        sentinel            = '',
        sentinel_semantic   = 'отсутствует',
        semantic            = 'Триггеры перехода (JSON-массив строк, например \'["branch:to:Elon"]\'). "" - отсутствует.',
        computation         = 'aef.metadata.langgraph_triggers; иначе ""',
        validation          = ((pl.col('meta_langgraph_triggers') == '') |
                                    pl.col('meta_langgraph_triggers').str.contains(re_json_str_array)),
    ),
    SpanDataAttr(
        group               = 'Метаданные LangGraph',
        name                = 'meta_langgraph_path',
        parquet_type        = 'BYTE_ARRAY (UTF8)',
        polars_type         = pl.String,
        constraints         = f'NOT NULL, "" если отсутствует, иначе строка, соответствующая регулярному выражению: {re_json_str_array}',
        sentinel            = '',
        sentinel_semantic   = 'отсутствует',
        semantic            = 'Маршрут выполнения в графе (JSON-массив строк, например \'["__pregel_pull","Elon"]\'). "" - отсутствует.',
        computation         = 'aef.metadata.langgraph_path; иначе ""',
        validation          = ((pl.col('meta_langgraph_path') == '') |
                                    pl.col('meta_langgraph_path').str.contains(re_json_str_array)),
    ),
    SpanDataAttr(
        group               = 'Метаданные LangGraph',
        name                = 'meta_checkpoint_ns',
        parquet_type        = 'BYTE_ARRAY (UTF8)',
        polars_type         = pl.String,
        constraints         = 'NOT NULL, "" если отсутствует',
        sentinel            = '',
        sentinel_semantic   = 'отсутствует',
        semantic            = 'Пространство имён чекпоинта LangGraph. "" - отсутствует.',
        computation         = 'aef.metadata.langgraph_checkpoint_ns (приоритетно); при отсутствии — aef.metadata.checkpoint_ns; иначе ""',
        validation          = pl.lit(True),
    ),
    SpanDataAttr(
        group               = 'Метаданные LangGraph',
        name                = 'meta_tags',
        parquet_type        = 'BYTE_ARRAY (UTF8)',
        polars_type         = pl.String,
        constraints         = f'NOT NULL, "" если отсутствует, иначе строка, соответствующая регулярному выражению: {re_json_str_array}',
        sentinel            = '',
        sentinel_semantic   = 'отсутствует',
        semantic            = 'Теги (JSON-массив строк, например \'["graph:step:1"]\'). "" - отсутствует.',
        computation         = 'aef.metadata.tags; иначе ""',
        validation          = ((pl.col('meta_tags') == '') |
                                    pl.col('meta_tags').str.contains(re_json_str_array)),
    ),
    SpanDataAttr(
        group               = 'Метаданные LangGraph',
        name                = 'meta_extra',
        parquet_type        = 'BYTE_ARRAY (UTF8)',
        polars_type         = pl.String,
        constraints         = 'NOT NULL, "" если нет дополнительных полей, иначе требуется валидная JSON-строка',
        sentinel            = '',
        sentinel_semantic   = 'дополнительные поля отсутствуют',
        semantic            = 'Остальные поля aef.metadata, не выделенные в отдельные колонки (JSON-объект). "" - отсутствуют.',
        computation         = 'Оставшиеся ключи aef.metadata после извлечения известных; если нет -> ""',
        validation          = ((pl.col('meta_extra') == '') |
                                    pl.col('meta_extra').map_elements(is_empty_or_valid_json, return_dtype = pl.Boolean)),
    ),
    SpanDataAttr(
        group               = 'Служебные',
        name                = 'service_name',
        parquet_type        = 'BYTE_ARRAY (UTF8)',
        polars_type         = pl.String,
        constraints         = 'NOT NULL, "" если не указан',
        sentinel            = '',
        sentinel_semantic   = 'не задано',
        semantic            = 'Имя сервиса, сгенерировавшего телеметрию. "" - не задано.',
        computation         = 'resource_spans[].resource.attributes."service.name"; если отсутствует -> ""',
        validation          = pl.lit(True),
        is_mandatory        = False,
    ),
    SpanDataAttr(
        group               = 'Служебные',
        name                = 'service_version',
        parquet_type        = 'BYTE_ARRAY (UTF8)',
        polars_type         = pl.String,
        constraints         = 'NOT NULL, "" если не указана',
        sentinel            = '',
        sentinel_semantic   = 'не задана',
        semantic            = 'Версия сервиса. "" - не задана.',
        computation         = 'resource_spans[].resource.attributes."service.version"; если отсутствует -> ""',
        validation          = pl.lit(True),
        is_mandatory        = False,
    ),
    SpanDataAttr(
        group               = 'Мета-атрибуты',
        name                = 'session_id_derived',
        parquet_type        = 'BOOLEAN',
        polars_type         = pl.Boolean,
        constraints         = 'NOT NULL',
        sentinel            = None,
        sentinel_semantic   = None,
        semantic            = 'Признак того, что session_id был получен от родительского/дочернего спана в трассе',
        computation         = 'Вычисляется согласно описанным ниже правилам',
        validation          = ~(pl.col('session_id_derived') & pl.col('session_id_generated')),
        is_mandatory        = True,
    ),
    SpanDataAttr(
        group               = 'Мета-атрибуты',
        name                = 'session_id_generated',
        parquet_type        = 'BOOLEAN',
        polars_type         = pl.Boolean,
        constraints         = 'NOT NULL',
        sentinel            = None,
        sentinel_semantic   = None,
        semantic            = 'Признак того, что session_id содержит сгенерированное случайное значение',
        computation         = 'Вычисляется согласно описанным ниже правилам',
        validation          = ~(pl.col('session_id_derived') & pl.col('session_id_generated')),
        is_mandatory        = True,
    ),
    SpanDataAttr(
        group               = 'Служебные',
        name                = 'nexus_distrib_ver',
        parquet_type        = 'BYTE_ARRAY (UTF8)',
        polars_type         = pl.String,
        constraints         = 'NOT NULL, Строка в формате "X.Y.Z", где X, Y и Z -- неотрицательные целые числа, либо символ "?", если данный элемент версии не известен',
        sentinel            = None,
        sentinel_semantic   = None,
        semantic            = 'Версия дистрибутива Nexus.',
        computation         = 'если отсутствует -> "?.?.?"',
        validation          = pl.col('nexus_distrib_ver').str.contains(r'^(\d+|\?)\.(\d+|\?)\.(\d+|\?)$'),
        is_mandatory        = False,
    ),
))

# ============================================================
# ГЕНЕРАЦИЯ СПЕЦИФИКАЦИИ
# ============================================================

specification = pl.DataFrame(
    map(lambda attr: tuple(map(str, (attr.group, attr.is_mandatory, attr.name, attr.parquet_type, attr.polars_type, attr.constraints, attr.sentinel, attr.sentinel_semantic, attr.semantic, attr.computation))),
        spec.attrs),
    schema = (
        'группа атрибутов',
        'обязательность',
        'атрибут',
        'тип Parquet',
        'тип Polars',
        'ограничения',
        'страж',
        'семантика стража',
        'семантика атрибута',
        'вычисление (предположительно)',
    ),
    orient = 'row'
).with_row_index('#', offset = 1).select(pl.col('группа атрибутов', '#'), pl.exclude('группа атрибутов', '#'))

print('спецификация таблицы spans:')
with pl.Config(tbl_rows = 100, tbl_cols = 100, tbl_width_chars = 10000, fmt_str_lengths = 1000) as _:
    print(specification)



# ============================================================
# ПОЛУЧЕНИЕ СХЕМЫ ЗАГРУЗКИ
# ============================================================

load_schema = pl.Schema(
    map(lambda attr: (attr.name, attr.polars_type), spec.attrs),
    check_dtypes = True
)

# ============================================================
# ЗАГРУЗКА ДАННЫХ
# ============================================================

loaded_spans = pl.scan_parquet(
    parquet_path.as_posix(),
    schema          = load_schema,
    rechunk         = True,
    missing_columns = 'raise',
    extra_columns   = 'raise',
)

with pl.Config(tbl_rows = 100, tbl_cols = 100, tbl_width_chars = 10000, fmt_str_lengths = 1000) as _:
    print(ls_data := loaded_spans.with_row_index(offset = 1).collect())

# ============================================================
# ВАЛИДАЦИЯ ДАННЫХ
# ============================================================

validation = map(lambda attr: (attr.name, attr.validation), spec.attrs)
validated_spans = loaded_spans.with_columns(
    starmap(lambda c, e: e.alias(f'{c}_validated'), validation)
).collect()

duplicated_spans = (
    loaded_spans
    .group_by('trace_id', 'span_id')
    .len()
    .filter(pl.col('len') > 1)
)
span_id_is_unique_in_trace = duplicated_spans.collect().is_empty()

validation_stats = (
    pl.DataFrame({
        'column':   pl.Series(load_schema.keys(), dtype = pl.String),
        'valid':    validated_spans.select(pl.col('^.+_validated$').sum()).row(0),
        'total':    validated_spans.height,
    })
    .with_columns(
        pl.col('total').sub(pl.col('valid')).alias('invalid'),
        pl.col('valid').truediv(pl.col('total')).mul(100).round(2).alias('valid_pct'),
    )
    .select('column', 'total', 'valid', 'invalid', 'valid_pct')
    .sort('valid_pct')
)
all_fields_valid = validation_stats.filter(pl.col('valid_pct').cast(pl.UInt8).ne(100)).is_empty()

print('статистика валидации')
with pl.Config(tbl_rows = 100, tbl_cols = 100, tbl_width_chars = 10000, fmt_str_lengths = 1000) as _:
    print(validation_stats)


assert span_id_is_unique_in_trace and all_fields_valid, 'данные не прошли валидацию'
print('данные валидны')


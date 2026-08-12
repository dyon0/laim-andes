from    typing      import ClassVar, Literal
from    dataclasses import dataclass


type PhysType   = Literal['BYTE_ARRAY (UTF8)', 'INT64', 'FLOAT', 'BOOLEAN']
type AttrGroup  = Literal[
    'id', 'time', 'status', 'classification', 'text_data',
    'LLM', 'HTTP', 'Kafka', 'LangGraph_meta', 'service', 'meta_attrs']


@dataclass(frozen = True)
class Sentinel:
    unknown : ClassVar[str] = 'unknown'
    trash   : ClassVar[str] = 'trash'
    root    : ClassVar[str] = 'root'
    outside : ClassVar[str] = 'outside'
    unset   : ClassVar[str] = 'unset'
    none    : ClassVar[str] = 'none'

    token   : ClassVar[str] = 'TRASH'

    empties : ClassVar[dict] = {
        'INT64'             : -1,
        'FLOAT'             : -1.0,
        'BOOLEAN'           : False,
        'BYTE_ARRAY (UTF8)' : ''}

    trashes : ClassVar[dict] = {
        'INT64'             : -2,
        'FLOAT'             : -2.0,
        'BYTE_ARRAY (UTF8)' : token}

    domains : ClassVar[dict] = {
        'root'      : 'root',
        'outside'   : 'outside',
        'unset'     : 'STATUS_CODE_UNSET',
        'none'      : 'NONE'}

    @staticmethod
    def empty(role: None | str, phys: PhysType) -> None | bool | int | float | str:
        return (None                            if role is None
                else Sentinel.empties.get(phys) if role == Sentinel.unknown
                else Sentinel.domains.get(role))

    @staticmethod
    def garbage(phys: PhysType) -> None | int | float | str:
        return Sentinel.trashes.get(phys)


@dataclass(frozen = True)
class Patterns:
    base64          : ClassVar[str] = r'^[A-Za-z0-9+/]+={0,2}$'
    codebase_elem   : ClassVar[str] = r'^C[IE][0-9]+$'
    nexus_version   : ClassVar[str] = r'^(\d+|\?)\.(\d+|\?)\.(\d+|\?)$'
    json_str_array  : ClassVar[str] = r'^\[\s*"([^"\\]|\\.)*"(\s*,\s*"([^"\\]|\\.)*")*\s*\]$'
    json_like       : ClassVar[str] = r'^\s*(\{[\s\S]*\}|\[[\s\S]*\])\s*$'


@dataclass(frozen = True)
class Categories:
    status_code     : ClassVar[tuple[str, ...]] = (
        'STATUS_CODE_UNSET', 'STATUS_CODE_OK', 'STATUS_CODE_ERROR')

    aef_kind        : ClassVar[tuple[str, ...]] = (
        'llm', 'start_agent', 'chain', 'tool', 'retriever', 'input_request',
        'output_request', 'kafka_produce', 'kafka_consume', 'other', 'guard')

    http_method     : ClassVar[tuple[str, ...]] = (
        'GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS')
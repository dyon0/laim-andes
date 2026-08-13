from    typing                  import Literal
from    dataclasses             import dataclass, field

from    pathlib                 import PurePath

from    ars.specification.spec  import DataObject
from    ars.tools.tui.tui       import ColorSchemeDataScience


@dataclass(frozen = True)
class FeatureParams:
    error                       : str                                   = r'(?i)\b(error|exception|fail|ошибка)\b'
    sentence                    : str                                   = r'[.!?]+'
    punctuation                 : str                                   = r'[.,!?:;–—()\[\]{}«»\'\']'
    uppercase                   : str                                   = r'[А-ЯA-Z]'
    special_char                : str                                   = r'[^а-яА-Яa-zA-Z0-9\s]'
    digit                       : str                                   = r'\d'

    quantiles                   : tuple[float, ...]                     = (0.25, 0.50, 0.75, 0.90, 0.95)
    rolling_windows             : tuple[int, ...]                       = (3, 5, 10, 30, 50, 100, 500, 1000)

    eps                         : float                                 = 1e-8


@dataclass(frozen = True)
class S1Config:
    input_parquet_files         : tuple[PurePath, ...]
    output_dir                  : PurePath
    output_prefix               : str

    #embedding_cache             : PurePath                              = PurePath('/mnt/data/models/embedders/encodechka/')
    #embedding_model             : str                                   = 'deepvk/USER-bge-m3'
    embedder_path               : PurePath                              = PurePath('/mnt/data/models/embedder/encodechka.zip')
    embedding_batch_size        : int                                   = 32
    embedding_max_length        : int                                   = 1024
    embedding_gpus              : int                                   = 0     # 0 = все видимые GPU; N = первые N (только при device=cuda)
    embedding_pool_chunk        : int                                   = 5000  # текстов на воркер за одну раздачу пула
    device                      : str                                   = 'cpu'

    fill_values                 : dict[float | str, tuple[str, ...]]    = field(
        default_factory = lambda: {'NonAnomaly': ('anomaly_type',)})
    default_fill_value          : float | str                           = 0.0
    fill_group_col              : dict[str, str]                        = field(default_factory = dict)

    min_fill_rate               : float                                 = 0.500
    max_static_rate             : float                                 = 0.950
    max_correlation             : float                                 = 0.999

    use_meta_sem                : bool                                  = False
    meta_sem_template           : str                                   = 'Агент {agent_id} выполнил {kind} длительностью {duration:.2f} сек. Характеристики: {details}'
    use_llm_sem                 : bool                                  = False
    llm_sem_use_stub            : bool                                  = True
    llm_sem_model               : str                                   = 'deepvk/USER-bge-m3'
    llm_sem_prompt_template     : str                                   = 'Опиши кратко на русском языке следующее событие в мультиагентной системе:\nАгент: {agent_id};\nТип: {kind};\nДлительность: {duration:.2f} сек.;\nХарактеристики: {details}\n'
    llm_sem_max_new_tokens      : int                                   = 64
    llm_sem_temperature         : float                                 = 0.7
    llm_sem_batch_size          : int                                   = 8
    llm_sem_do_sample           : bool                                  = True

    export_features             : bool                                  = True
    inject_anomalies            : bool                                  = True

    norm_train_ratio            : float                                 = 0.70
    norm_val_ratio              : float                                 = 0.15
    anom_val_ratio              : float                                 = 0.70
    anom_test_ratio             : float                                 = 0.30

    epi_normalization           : Literal['zscore', 'robust']           = 'robust'
    winsorize_epi               : bool                                  = True
    winsorize_limits            : tuple[float, float]                   = (0.01, 0.99)
    # F-05: floor for the robust scale (IQR/std) and symmetric clip for
    # normalized values; 0.0 disables either (legacy behavior).
    scale_floor                 : float                                 = 1e-2
    norm_z_clip                 : float                                 = 20.0
    max_abs_feature             : float                                 = 1e6

    recast                      : bool                                  = False
    eps_normalization           : float                                 = 1e-6

    eps_divide                  : float                                 = 1e-8
    duration_scale_to_sec       : float                                 = 1e9
    default_agent_id            : str                                   = 'CI1'

    data_object                 : DataObject                            = field(default_factory = DataObject)
    feature_params              : FeatureParams                         = field(default_factory = FeatureParams)

    output_color_scheme         : ColorSchemeDataScience                = field(default_factory = ColorSchemeDataScience)
    output_max_collection_len   : None | int                            = 10

    samples_dir                 : None | PurePath                       = None #PurePath('/mnt/data/traces/processed/samples')

    seed_random                 : int                                   = 12345
    seed_polars                 : int                                   = 12345
    seed_torch                  : int                                   = 12345
    seed_split                  : int                                   = 12345
    seed_synth                  : int                                   = 12345
    seed_llm                    : int                                   = 12345
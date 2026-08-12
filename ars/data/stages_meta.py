from    typing      import Literal, Tuple, Dict
from    dataclasses import dataclass


#todo: str -> PurePath везде, где необходимо; явно приводить к str (через настраиваемый метод) только где это необходимо, например, для сериализации Meta в json
@dataclass(frozen = True)
class S1Meta:
    '''артефакты подготовки данных, передаюся между этапами'''

    prefix              : str
    run_id              : None | str
    raw_files           : Tuple[str, ...]
    output_dir          : str
    seed_polars         : int
    seed_random         : int
    seed_torch          : int
    seed_split          : int
    seed_synth          : int
    seed_llm            : int
    embedding_model     : str
    epi_dim             : int
    epi_features        : Tuple[str, ...]
    epi_normalization   : Dict[str, Literal['zscore', 'robust'] | bool | float | Tuple[float, float]]
    anomaly_types       : Tuple[str, ...]
    semantic_vectors    : Dict[str, int]
    #synth_anom_injected     : None | Dict[str, Any]
    split_config    : Dict[str, float]

    train_samples       : int
    val_samples         : int
    test_samples        : int
    train_normal_count  : int
    train_anomaly_count : int
    val_normal_count    : int
    val_anomaly_count   : int
    test_normal_count   : int
    test_anomaly_count  : int


@dataclass(frozen = True)
class S2Meta:
    '''артефакты обучения детектора, передаюся между этапами'''

    output_dir          : str
    experiment_dir      : str
    best_experiment     : str
    select_metric       : str
    best_metric_value   : float

    max_len         : int
    epi_dim         : int
    sem_dim         : int
    epi_sz_latent   : int
    sem_sz_latent   : int
    seq_pad_chunk   : int

    best_threshold      : float
    normalize_latent    : bool

    epi_latent_mean : None | Tuple[float, ...]
    epi_latent_std  : None | Tuple[float, ...]
    sem_latent_mean : None | Tuple[float, ...]
    sem_latent_std  : None | Tuple[float, ...]

    test_metrics    : Dict[str, float]
    calibration     : Dict[str, float]


@dataclass(frozen = True)
class S3Meta:
    '''артефакты обучения классификатора типов аномалий, передаюся между этапами'''

    output_dir          : str
    experiment_dir      : str
    best_experiment     : str
    select_metric       : str
    best_metric_value   : float

    n_classes       : int
    class_names     : Tuple[str, ...]
    feature_dim     : int
    feature_layout  : Dict[str, int]

    base_kinds  : Tuple[str, ...]
    meta_solver : str
    n_folds     : int

    test_metrics    : Dict[str, float]
    metaparams      : Tuple[float, ...]

    s2_meta : S2Meta
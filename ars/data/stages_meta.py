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

    # F-10: fingerprint of the embedding model the artifacts were built with;
    # None only for legacy artifacts predating the field
    embedding_fingerprint : None | str = None

    # F-79: per-class injection coverage (planned / no_victims / labeled /
    # applied / unapplied — see anomalies_injection.injection_coverage);
    # unapplied traces are excluded from val/test. None: no injection or a
    # legacy artifact
    injection_coverage : None | Dict[str, Dict[str, float]] = None


@dataclass(frozen = True)
class S2Meta:
    '''артефакты обучения детектора, передаюся между этапами'''

    output_dir          : str
    experiment_dir      : str
    best_experiment     : str
    select_metric       : str
    # F-83: the experiment is chosen on `select_on` (VAL since F-02);
    # selection_value is select_metric THERE — what the choice was based on —
    # and test_value the same metric on TEST (reporting only). The old single
    # `best_metric_value` held the TEST value; legacy JSON is read by
    # from_dict (selection_value None -> unknown).
    select_on           : str
    selection_value     : None | float
    test_value          : float

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

    # rows of every inference-style forward call of the training run
    # (PreparedData.infer_rows). Eval, scoring, attribution and s3 reuse it:
    # on GPU a forward whose shapes were already compiled/autotuned in the
    # process compiles in ~2 s, a new shape in up to ~20 s. None: legacy
    # artifact — each call then uses its input's own bucket (block_rows)
    infer_rows      : None | int = None

    @classmethod
    def from_dict(cls, raw: dict) -> 'S2Meta':
        '''s2_meta.json -> S2Meta, including artifacts written before F-83'''
        data = _legacy_selection(dict(raw))
        for k in ('epi_latent_mean', 'epi_latent_std', 'sem_latent_mean', 'sem_latent_std'):
            data[k] = tuple(data[k]) if data.get(k) is not None else None
        return cls(**data)

    @property
    def selection_value_or_legacy(self) -> float:
        '''the value s3 used as a constant metaparameter: selection_value, or —
        for a pre-F-83 artifact — the TEST value it was trained with'''
        return self.selection_value if self.selection_value is not None else self.test_value


@dataclass(frozen = True)
class S3Meta:
    '''артефакты обучения классификатора типов аномалий, передаюся между этапами'''

    output_dir          : str
    experiment_dir      : str
    best_experiment     : str
    select_metric       : str
    # F-83: see S2Meta — selection_value on select_on (VAL), test_value on TEST
    select_on           : str
    selection_value     : None | float
    test_value          : float

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

    @classmethod
    def from_dict(cls, raw: dict) -> 'S3Meta':
        '''s3_meta.json -> S3Meta, including artifacts written before F-83'''
        data = _legacy_selection(dict(raw))
        data['s2_meta'] = S2Meta.from_dict(data['s2_meta'])
        for k in ('class_names', 'base_kinds', 'metaparams'):
            data[k] = tuple(data[k])
        data['feature_layout'] = dict(data['feature_layout'])
        return cls(**data)


def _legacy_selection(data: dict) -> dict:
    '''F-83: before the fix a single `best_metric_value` held the TEST value of
    select_metric; the value on the selection split was not stored.'''
    if 'best_metric_value' in data:
        data.setdefault('test_value', data.pop('best_metric_value'))
        data.setdefault('selection_value', None)
        data.setdefault('select_on', 'val')
    return data
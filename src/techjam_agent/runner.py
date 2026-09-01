from __future__ import annotations
import csv
import copy
import hashlib
import importlib.util
import json
import os
import pickle
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any
import numpy as np
import torch
from .bpr import bpr_step, build_group_softmax_indices, build_pair_indices
from .autonomous_branch import AutonomousRuntime, load_code_branch
from .config import normalize_config, validate_config
from .dcnv2 import DCNv2
from .error_slices import build_error_slice_report
from .deepfm import DeepFM
from .field_aware import FieldAwareFM
from .history_features import AFFINITY_FEATURES, TrainHistoryStatistics
from .history_models import CandidateAwareDIN, MetadataSASRec
from .lightgcn import LightGCN
from .model_interface import recommender_operator, registered_model_ids
from .multitask import MultiTaskRecommender
from .sasrec import SASRecCandidateScorer
from .sequence import SequenceContext, build_previous_positive_context
from .sequential import FPMC, SequentialFM
from .two_tower import TwoTowerCandidateModel
DURATION_FEATURE = 'duration_fine_bucket'
TAG_FEATURE = 'tag'
TAG_INDEX = 7
RAW_CATEGORICAL_INDICES = {
    TAG_FEATURE: 7,
    'hour': 9,
    'weekday': 10,
    'video_type': 12,
    'user_activity': 13,
}
RAW_CATEGORICAL_FEATURES = set(RAW_CATEGORICAL_INDICES)
UPLOAD_AGE_INDEX = 11
TEMPORAL_POPULARITY_INDICES = {
    'time_decay_item_popularity': 2,
    'time_decay_author_popularity': 3,
    'time_decay_tag_popularity': 7,
}
DERIVED_FEATURES = {
    'upload_age_bucket', 'freshness_decay', 'recent_history_similarity',
    *TEMPORAL_POPULARITY_INDICES,
}
SPLIT_DATES = {'train': (20220408, 20220421), 'valid': (20220422, 20220428), 'test': (20220429, 20220508)}
STANDARD_LOG_FILES = ('log_standard_4_08_to_4_21_pure.csv', 'log_standard_4_22_to_5_08_pure.csv')
CACHE_SCHEMA_VERSION = 'v2'

def _duration_edges(rows: list[tuple], bins: int=50) -> np.ndarray:
    durations = np.asarray([row[5] for row in rows], dtype=np.float64)
    return np.quantile(durations, np.linspace(0.0, 1.0, bins + 1)[1:-1])

def _ranking_order_and_groups(users) -> tuple[np.ndarray, list[int]]:
    values = np.asarray(users)
    order = np.argsort(values, kind='stable')
    sorted_users = values[order]
    if len(sorted_users) == 0:
        return (order, [])
    boundaries = np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1
    groups = np.diff(np.concatenate(([0], boundaries, [len(sorted_users)])))
    return (order, [int(value) for value in groups])

def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f'cannot import {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def _binary(value: Any) -> float:
    return 0.0 if value in (None, '', '0', 0, 0.0, False) else 1.0

class ExperimentRunner:

    def __init__(self, root: Path, data_dir: Path, starter_dir: Path, evaluator_sha256: str | None=None) -> None:
        self.root = root
        self.data_dir = data_dir
        self.starter_dir = starter_dir
        self.evaluator_sha256 = evaluator_sha256
        sys.path.insert(0, str(starter_dir))
        self.data = _load_module('techjam_starter_data', starter_dir / 'data.py')
        self.evaluate_mod = _load_module('techjam_starter_evaluate', starter_dir / 'evaluate.py')
        self.baseline = _load_module('techjam_starter_baseline', starter_dir / 'baseline.py')
        self.submit = _load_module('techjam_starter_submit', starter_dir / 'submit.py')
        self._splits = None
        self._encoded = None
        self._sequence_context: SequenceContext | None = None
        self._categorical_cache: dict[str, tuple[dict[str, np.ndarray], int]] = {}
        self._raw_sidecar_cache: dict[str, dict[str, np.ndarray]] | None = None
        self._sasrec_cache: dict[int, tuple[dict[str, dict[str, np.ndarray]], int]] = {}
        self._metadata_history_cache: dict[int, tuple[dict[str, dict[str, np.ndarray]], dict[str, int]]] = {}
        self.cache_dir = self.root / 'artifacts' / 'cache'
        self._dataset_fingerprint: str | None = None

    def verify_evaluator(self) -> None:
        if not self.evaluator_sha256:
            return
        path = self.starter_dir / 'evaluate.py'
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != self.evaluator_sha256:
            raise RuntimeError(f'official evaluator integrity check failed: {actual}')

    def prepare(self) -> None:
        self.verify_evaluator()
        required = ('video_features_basic_pure.csv', 'user_features_pure.csv', *STANDARD_LOG_FILES)
        missing = [name for name in required if not (self.data_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(f"dataset is missing from {self.data_dir}: {', '.join(missing)}")
        digest = hashlib.sha256()
        for name in required:
            stat = (self.data_dir / name).stat()
            digest.update(f'{name}:{stat.st_size}:{stat.st_mtime_ns}'.encode())
        digest.update((self.starter_dir / 'data.py').read_bytes())
        self._dataset_fingerprint = digest.hexdigest()[:20]
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self.cache_dir / f'dataset_{self._dataset_fingerprint}.pkl'
        if cache_path.is_file():
            print(f'Loading encoded dataset cache {cache_path.name} ...', flush=True)
            with cache_path.open('rb') as handle:
                self._splits, self._encoded = pickle.load(handle)
        else:
            print(f'Loading dataset from {self.data_dir} ...', flush=True)
            self._splits = self.data.load(str(self.data_dir))
            self._encoded = self.data.encode(self._splits)
            temporary = cache_path.with_suffix(f'.{os.getpid()}.tmp')
            with temporary.open('wb') as handle:
                pickle.dump((self._splits, self._encoded), handle, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(temporary, cache_path)
        print('Dataset ready: ' + ', '.join((f'{name}={len(rows):,}' for name, rows in self._splits.items())), flush=True)

    def run(self, config: dict[str, Any], checkpoint: Path) -> dict[str, Any]:
        config = normalize_config(config)
        validate_config(config)
        if self._encoded is None:
            self.prepare()
        return recommender_operator(config['model']).fit_validate(self, config, checkpoint)

    def finalize(self, config: dict[str, Any], checkpoint: Path, output: Path) -> dict[str, Any]:
        config = normalize_config(config)
        validate_config(config)
        if self._encoded is None:
            self.prepare()
        return recommender_operator(config['model']).finalize(self, config, checkpoint, output)

    # Public, deliberately small adapter surface for generated branches. A
    # branch can either compose a built-in scorer or train its own model using
    # encoded matrices and the official evaluator, without reaching into the
    # Controller or data files.
    def autonomous_encoded(self, config: dict[str, Any], split: str = "train_valid"):
        if self._encoded is None:
            self.prepare()
        encoded, width = self._encoded_for(normalize_config(config))
        if split == "train_valid":
            return ({name: encoded[name] for name in ("train", "valid")}, width)
        if split == "test":
            X, _, users = encoded["test"]
            return ({"test": (X, None, users)}, width)
        raise ValueError("split must be 'train_valid' or 'test'")

    def autonomous_dense_matrices(self, config: dict[str, Any], split: str = "train_valid"):
        if self._encoded is None:
            self.prepare()
        encoded, field_dims, numeric = self._dense_neural_data(normalize_config(config))
        if split == "train_valid":
            names = ("train", "valid")
        elif split == "test":
            names = ("test",)
        else:
            raise ValueError("split must be 'train_valid' or 'test'")
        safe_encoded = {
            name: (encoded[name][0], None if name == "test" else encoded[name][1], encoded[name][2])
            for name in names
        }
        safe_numeric = {name: numeric[name] for name in names}
        return safe_encoded, field_dims, safe_numeric

    def autonomous_evaluate(self, users: Any, labels: Any, scores: Any) -> dict[str, Any]:
        metrics = self.evaluate_mod.evaluate(users, labels, np.asarray(scores))
        return self._metrics(metrics)

    def autonomous_write_validation_slices(self, checkpoint: Path, scores: Any) -> None:
        self._write_validation_slices(checkpoint, np.asarray(scores))

    def autonomous_write_submission(self, scores: Any, output: Path) -> dict[str, Any]:
        return self._write_final(np.asarray(scores), output)

    @staticmethod
    def autonomous_save_checkpoint(checkpoint: Path, payload: Any) -> None:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, checkpoint)

    @staticmethod
    def autonomous_load_checkpoint(checkpoint: Path) -> Any:
        return torch.load(checkpoint, map_location="cpu", weights_only=False)

    def autonomous_run_builtin(
        self, model_name: str, config: dict[str, Any], checkpoint: Path
    ) -> dict[str, Any]:
        """Run one registered family as a warm-start/composition primitive."""
        allowed = set(registered_model_ids()) - {"custom"}
        if model_name not in allowed:
            raise ValueError(f"generated branch cannot delegate to model {model_name!r}")
        local = copy.deepcopy(normalize_config(config))
        local["model"] = model_name
        local.pop("code_branch", None)
        local.pop("code_branch_sha256", None)
        validate_config(local)
        return recommender_operator(model_name).fit_validate(self, local, checkpoint)

    def autonomous_finalize_builtin(
        self, model_name: str, config: dict[str, Any], checkpoint: Path, output: Path
    ) -> dict[str, Any]:
        allowed = set(registered_model_ids()) - {"custom"}
        if model_name not in allowed:
            raise ValueError(f"generated branch cannot delegate to model {model_name!r}")
        local = copy.deepcopy(normalize_config(config))
        local["model"] = model_name
        local.pop("code_branch", None)
        local.pop("code_branch_sha256", None)
        validate_config(local)
        return recommender_operator(model_name).finalize(self, local, checkpoint, output)

    @staticmethod
    def _custom_result(value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise RuntimeError("generated fit_validate/finalize must return a metrics object")
        required = ("GAUC", "nDCG@5", "primary")
        if any(key not in value for key in required):
            raise RuntimeError("generated branch result must contain GAUC, nDCG@5, and primary")
        cleaned = dict(value)
        for key in required:
            try:
                number = float(cleaned[key])
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"generated branch metric {key!r} is not numeric") from exc
            if not np.isfinite(number):
                raise RuntimeError(f"generated branch metric {key!r} is not finite")
            cleaned[key] = number
        return cleaned

    def _custom_module(self, config: dict[str, Any]):
        branch = config.get("code_branch")
        if not isinstance(branch, str) or not branch.strip():
            raise RuntimeError("custom model requires code_branch")
        return load_code_branch(self.root, branch, config.get("code_branch_sha256"))

    def _run_custom(self, config: dict[str, Any], checkpoint: Path) -> dict[str, Any]:
        module = self._custom_module(config)
        return self._custom_result(
            module.fit_validate(AutonomousRuntime(self), config, checkpoint)
        )

    def _finalize_custom(self, config: dict[str, Any], checkpoint: Path, output: Path) -> dict[str, Any]:
        module = self._custom_module(config)
        return self._custom_result(
            module.finalize(AutonomousRuntime(self, allow_finalize=True), config, checkpoint, output)
        )

    @staticmethod
    def _metrics(metrics: dict[str, Any]) -> dict[str, Any]:
        return {key: int(value) if key in ('users', 'rows') else float(value) for key, value in metrics.items()}

    @staticmethod
    def _validation_score(metrics: dict[str, Any], config: dict[str, Any]) -> float:
        """Return the metric used for training-time model selection.

        Promotion is deliberately still decided by Controller on the official
        ``primary`` score.  This hook only controls which validation checkpoint
        (or blend weights) is retained, so an nDCG-focused experiment can keep
        the epoch that actually improves top-k ordering instead of an epoch
        that happens to maximize GAUC.
        """
        metric = str(config.get("hyperparameters", {}).get("validation_metric", "primary"))
        if metric not in {"primary", "nDCG@5", "GAUC"}:
            metric = "primary"
        return float(metrics[metric])

    def _write_final(self, scores: np.ndarray, output: Path) -> dict[str, Any]:
        if not np.all(np.isfinite(scores)):
            raise RuntimeError('final predictions contain NaN/Inf')
        _, ytest, users = self._encoded[0]['test']
        metrics = self.evaluate_mod.evaluate(users, ytest, scores)
        output.parent.mkdir(parents=True, exist_ok=True)
        self.submit.write_submission(output, self._splits['test'], scores)
        self.submit.read_submission(output, self._splits['test'])
        return self._metrics(metrics)

    def _write_validation_slices(self, checkpoint: Path, scores: np.ndarray) -> None:
        report = build_error_slice_report(
            self._splits['train'], self._splits['valid'], np.asarray(scores),
            self.evaluate_mod.evaluate,
        )
        path = checkpoint.with_suffix('.slices.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f'.{os.getpid()}.tmp')
        temporary.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        os.replace(temporary, path)

    def _raw_categorical(self, feature: str) -> tuple[dict[str, np.ndarray], int]:
        cached = self._categorical_cache.get(feature)
        if cached is not None:
            return cached
        if feature not in RAW_CATEGORICAL_INDICES:
            raise ValueError(f'unsupported raw categorical feature: {feature}')
        value_index = RAW_CATEGORICAL_INDICES[feature]
        if not self._splits['train'] or len(self._splits['train'][0]) <= value_index:
            raise RuntimeError(f'{feature} requested but augmented data row is unavailable')
        vocabulary: dict[str, int] = {}
        for row in self._splits['train']:
            value = str(row[value_index] if row[value_index] not in (None, '') else 'UNK')
            if value not in vocabulary:
                vocabulary[value] = len(vocabulary)
        unknown = len(vocabulary)
        encoded = {split: np.fromiter((vocabulary.get(str(row[value_index] if row[value_index] not in (None, '') else 'UNK'), unknown) for row in rows), dtype=np.int32, count=len(rows)) for split, rows in self._splits.items()}
        result = (encoded, len(vocabulary) + 1)
        self._categorical_cache[feature] = result
        return result

    @staticmethod
    def _history_values(statistics: TrainHistoryStatistics, feature: str, rows: list[tuple], split: str, *, categorical: bool) -> np.ndarray:
        if split == 'train' and feature in AFFINITY_FEATURES:
            values = statistics.chronological_affinity_values(feature, rows, categorical=categorical)
        else:
            getter = statistics.categorical_value if categorical else statistics.numeric_value
            values = [getter(feature, row, leave_one_out=split == 'train') for row in rows]
        return np.asarray(values, dtype=np.int32 if categorical else np.float32)

    def _feature_cache_path(self, feature: str, categorical: bool) -> Path:
        if not self._dataset_fingerprint:
            raise RuntimeError('prepare must be called before feature caching')
        mode = 'cat' if categorical else 'num'
        return self.cache_dir / f'feature_{CACHE_SCHEMA_VERSION}_{self._dataset_fingerprint}_{feature}_{mode}.npz'

    def _temporal_popularity(self, feature: str) -> dict[str, np.ndarray]:
        key_index = TEMPORAL_POPULARITY_INDICES[feature]
        half_life_ms = 7.0 * 24 * 60 * 60 * 1000
        state: dict[Any, tuple[float, int]] = {}
        result: dict[str, np.ndarray] = {}
        train = self._splits['train']
        values = np.zeros(len(train), dtype=np.float32)
        order = sorted(range(len(train)), key=lambda i: (int(train[i][8]), i))
        start = 0
        while start < len(order):
            timestamp = int(train[order[start]][8])
            end = start + 1
            while end < len(order) and int(train[order[end]][8]) == timestamp:
                end += 1
            for index in order[start:end]:
                key = train[index][key_index]
                count, previous = state.get(key, (0.0, timestamp))
                values[index] = np.log1p(count * (0.5 ** ((timestamp - previous) / half_life_ms)))
            for index in order[start:end]:
                key = train[index][key_index]
                count, previous = state.get(key, (0.0, timestamp))
                state[key] = (count * (0.5 ** ((timestamp - previous) / half_life_ms)) + 1.0, timestamp)
            start = end
        result['train'] = values
        for split in ('valid', 'test'):
            rows = self._splits[split]
            values = np.zeros(len(rows), dtype=np.float32)
            for index, row in enumerate(rows):
                timestamp, key = int(row[8]), row[key_index]
                count, previous = state.get(key, (0.0, timestamp))
                values[index] = np.log1p(count * (0.5 ** (max(0, timestamp - previous) / half_life_ms)))
            result[split] = values
        return result

    def _recent_similarity(self) -> dict[str, np.ndarray]:
        histories: dict[Any, deque[tuple[Any, Any]]] = defaultdict(lambda: deque(maxlen=20))
        result: dict[str, np.ndarray] = {}

        def score(row: tuple) -> float:
            history = histories[row[1]]
            if not history:
                return 0.0
            tag_matches = sum(tag == row[7] for tag, _ in history)
            author_matches = sum(author == row[3] for _, author in history)
            return (2.0 * tag_matches + author_matches) / (3.0 * len(history))

        train = self._splits['train']
        values = np.zeros(len(train), dtype=np.float32)
        order = sorted(range(len(train)), key=lambda i: (int(train[i][8]), i))
        start = 0
        while start < len(order):
            timestamp = int(train[order[start]][8])
            end = start + 1
            while end < len(order) and int(train[order[end]][8]) == timestamp:
                end += 1
            for index in order[start:end]:
                values[index] = score(train[index])
            for index in order[start:end]:
                row = train[index]
                if int(row[6]) == 1:
                    histories[row[1]].append((row[7], row[3]))
            start = end
        result['train'] = values
        for split in ('valid', 'test'):
            result[split] = np.fromiter(
                (score(row) for row in self._splits[split]), dtype=np.float32,
                count=len(self._splits[split]),
            )
        return result

    def _feature_values(self, feature: str, *, categorical: bool) -> tuple[dict[str, np.ndarray], int]:
        cache_path = self._feature_cache_path(feature, categorical)
        if cache_path.is_file():
            loaded = np.load(cache_path, allow_pickle=False)
            return ({split: loaded[split] for split in self._splits}, int(loaded['width'][0]))
        if feature in RAW_CATEGORICAL_FEATURES:
            raw, width = self._raw_categorical(feature)
            values = {split: array.astype(np.int32 if categorical else np.float32) for split, array in raw.items()}
        elif feature == DURATION_FEATURE:
            edges = _duration_edges(self._splits['train'])
            values = {split: np.searchsorted(edges, np.asarray([row[5] for row in rows])).astype(np.int32 if categorical else np.float32) for split, rows in self._splits.items()}
            width = 50
        elif feature == 'upload_age_bucket':
            edges = np.asarray([0, 1, 3, 7, 14, 30, 60, 90, 180, 365], dtype=np.float32)
            values = {}
            for split, rows in self._splits.items():
                ages = np.asarray([row[UPLOAD_AGE_INDEX] for row in rows], dtype=np.float32)
                bucketed = np.where(ages < 0, len(edges) + 1, np.searchsorted(edges, ages))
                values[split] = bucketed.astype(np.int32 if categorical else np.float32)
            width = len(edges) + 2
        elif feature == 'freshness_decay':
            numeric = {}
            for split, rows in self._splits.items():
                ages = np.asarray([row[UPLOAD_AGE_INDEX] for row in rows], dtype=np.float32)
                numeric[split] = np.where(ages < 0, 0.0, np.exp(-ages / 30.0)).astype(np.float32)
            values = {split: (np.minimum(19, (array * 20).astype(np.int32)) if categorical else array.astype(np.float32)) for split, array in numeric.items()}
            width = 20
        elif feature in TEMPORAL_POPULARITY_INDICES:
            numeric = self._temporal_popularity(feature)
            values = {split: (np.minimum(19, (array * 3).astype(np.int32)) if categorical else array) for split, array in numeric.items()}
            width = 20
        elif feature == 'recent_history_similarity':
            numeric = self._recent_similarity()
            values = {split: (np.minimum(19, (array * 20).astype(np.int32)) if categorical else array) for split, array in numeric.items()}
            width = 20
        else:
            statistics = TrainHistoryStatistics.build(self._splits['train'], [feature])
            values = {split: self._history_values(statistics, feature, rows, split, categorical=categorical) for split, rows in self._splits.items()}
            width = 20
        temporary = cache_path.with_suffix(f'.{os.getpid()}.tmp.npz')
        np.savez(temporary, **values, width=np.asarray([width], dtype=np.int32))
        os.replace(temporary, cache_path)
        return values, width

    def _encoded_for(self, config: dict[str, Any]):
        base, base_dim = self._encoded
        enabled = [key for key, value in config['features'].items() if value and key != 'continuous_history_stats']
        if not enabled:
            return (base, base_dim)
        columns: dict[str, list[np.ndarray]] = {}
        next_offset = base_dim
        for feature in enabled:
            feature_values, width = self._feature_values(feature, categorical=True)
            for split in self._splits:
                columns.setdefault(split, []).append(feature_values[split] + next_offset)
            next_offset += width
        encoded = {}
        for split, (X, y, users) in base.items():
            encoded[split] = (np.column_stack([X, *columns[split]]).astype(np.int32), y, users)
        return (encoded, next_offset)

    def _lightgbm_matrices(self, config: dict[str, Any]):
        base, _ = self._encoded
        field_mins = base['train'][0].min(axis=0)
        categorical = {split: (values[0] - field_mins).astype(np.int32) for split, values in base.items()}
        use_global = config['features']['continuous_history_stats']
        explicit = [key for key, value in config['features'].items() if value and key != 'continuous_history_stats']
        if not use_global and (not explicit):
            return categorical
        extra: dict[str, list[np.ndarray]] = {split: [] for split in self._splits}
        history_explicit = [feature for feature in explicit if feature not in DERIVED_FEATURES and feature != DURATION_FEATURE and feature not in RAW_CATEGORICAL_FEATURES]
        requested_history = [*history_explicit, *(('continuous_history_stats',) if use_global else ())]
        statistics = TrainHistoryStatistics.build(self._splits['train'], requested_history) if requested_history else None
        duration_edges = _duration_edges(self._splits['train']) if DURATION_FEATURE in explicit else None
        if use_global:
            for group_name, key_index in (('user', 1), ('item', 2)):
                for split, rows in self._splits.items():
                    rates = np.empty(len(rows), dtype=np.float32)
                    counts = np.empty(len(rows), dtype=np.float32)
                    for index, row in enumerate(rows):
                        positives, impressions = statistics.groups[group_name].get(row[key_index], [0, 0])
                        if split == 'train':
                            positives -= int(row[6])
                            impressions -= 1
                        prior_rate = statistics.global_prior(int(row[6]) if split == 'train' else None)
                        rates[index] = (positives + 20.0 * prior_rate) / (impressions + 20.0)
                        counts[index] = np.log1p(max(0, impressions))
                    extra[split].extend((rates, counts))
        for feature in explicit:
            values, _ = self._feature_values(feature, categorical=False)
            for split in self._splits:
                extra[split].append(values[split])
        return {split: np.column_stack([categorical[split], *extra[split]]).astype(np.float32) for split in self._splits}

    @staticmethod
    def _lightgbm_categorical_columns(config: dict[str, Any]) -> list[int]:
        columns = list(range(5))
        position = 5 + (4 if config['features']['continuous_history_stats'] else 0)
        for feature, enabled in config['features'].items():
            if not enabled or feature == 'continuous_history_stats':
                continue
            if feature in RAW_CATEGORICAL_FEATURES:
                columns.append(position)
            position += 1
        return columns

    def _new_latent_model(self, config: dict[str, Any], dim: int, fields: int, seed: int):
        hp = config['hyperparameters']
        if config['model'] == 'ffm':
            return FieldAwareFM(dim, fields, k=hp['embedding_dim'], lr=hp['learning_rate'], l2=hp['l2'], seed=seed)
        embedding_dim = 0 if config['model'] == 'linear' else hp['embedding_dim']
        return self.baseline.FM(dim, k=embedding_dim, lr=hp['learning_rate'], l2=hp['l2'], seed=seed)

    def _bpr_pairs(self, config: dict[str, Any], Xtr: np.ndarray, users, labels: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        hp = config['hyperparameters']
        strategy = hp['negative_sampling_strategy']
        match_values = {'random': None, 'same_tab': Xtr[:, 3], 'same_author': Xtr[:, 2]}[strategy]
        return build_pair_indices(users, labels, rng, hp['negatives_per_positive'], match_values)

    def _run_latent(self, config: dict[str, Any], checkpoint: Path) -> dict[str, Any]:
        enc, dim = self._encoded_for(config)
        hp = config['hyperparameters']
        Xtr, ytr, utr = enc['train']
        Xva, yva, uva = enc['valid']
        if config['model'] == 'fm_ensemble':
            member_seeds = list(range(hp['ensemble_size'])) if hp['ensemble_seed_set'] == 'sequential' else [int(v) for v in hp['ensemble_seed_set'].split(',')]
        else:
            member_seeds = [hp['seed']]
        started = time.monotonic()
        states = []
        best_epochs = []
        cumulative_scores = np.zeros(len(yva), dtype=np.float64)
        for member_number, seed in enumerate(member_seeds, start=1):
            model = self._new_latent_model(config, dim, Xtr.shape[1], seed)
            rng = np.random.default_rng(seed)
            best_score, best_state, bad, best_epoch = (-1.0, None, 0, 0)
            for epoch in range(1, hp['epochs'] + 1):
                if config['training_objective'] == 'bpr':
                    positive, negative = self._bpr_pairs(config, Xtr, utr, ytr, rng)
                    for start in range(0, len(positive), hp['batch_size']):
                        sl = slice(start, start + hp['batch_size'])
                        pos_x = Xtr[positive[sl]]
                        neg_x = Xtr[negative[sl]]
                        if config['model'] == 'ffm':
                            model.bpr_step(pos_x, neg_x)
                        else:
                            bpr_step(model, pos_x, neg_x)
                else:
                    indices = rng.permutation(len(ytr))
                    for start in range(0, len(indices), hp['batch_size']):
                        batch = indices[start:start + hp['batch_size']]
                        model.step(Xtr[batch], ytr[batch])
                metrics = self.evaluate_mod.evaluate(uva, yva, model.predict(Xva))
                current = self._validation_score(metrics, config)
                prefix = f'member {member_number}/{len(member_seeds)} | ' if len(member_seeds) > 1 else ''
                print(f"    {prefix}epoch {epoch:02d} | selection={config['hyperparameters'].get('validation_metric', 'primary')}:{current:.6f} | primary={float(metrics['primary']):.6f} | nDCG@5={float(metrics['nDCG@5']):.6f} | best={max(best_score, current):.6f}", flush=True)
                if current > best_score + 1e-05:
                    best_score, bad, best_epoch = (current, 0, epoch)
                    best_state = (model.V.copy(), model.W.copy(), np.float32(model.b))
                else:
                    bad += 1
                    if bad >= hp['patience']:
                        break
            if best_state is None:
                raise RuntimeError(f'training produced no checkpoint for seed {seed}')
            model.V, model.W, model.b = best_state
            states.append(best_state)
            best_epochs.append(best_epoch)
            cumulative_scores += model.predict(Xva)
        valid = self.evaluate_mod.evaluate(uva, yva, cumulative_scores / len(member_seeds))
        self._write_validation_slices(checkpoint, cumulative_scores / len(member_seeds))
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        if len(states) == 1:
            V, W, b = states[0]
            np.savez_compressed(checkpoint, V=V, W=W, b=b, best_epoch=np.asarray(best_epochs[0]))
        else:
            np.savez_compressed(checkpoint, V=np.stack([s[0] for s in states]), W=np.stack([s[1] for s in states]), b=np.asarray([s[2] for s in states]), best_epoch=np.asarray(best_epochs), seeds=np.asarray(member_seeds))
        return {'GAUC': float(valid['GAUC']), 'nDCG@5': float(valid['nDCG@5']), 'primary': float(valid['primary']), 'best_epoch': int(best_epochs[0]) if len(best_epochs) == 1 else best_epochs, 'ensemble_size': len(member_seeds), 'runtime_seconds': float(time.monotonic() - started)}

    def _finalize_latent(self, config: dict[str, Any], checkpoint: Path, output: Path) -> dict[str, Any]:
        enc, dim = self._encoded_for(config)
        Xtest = enc['test'][0]
        with np.load(checkpoint) as state:
            stored_v, stored_w, stored_b = (state['V'], state['W'], state['b'])
        single_state_ndim = 3 if config['model'] == 'ffm' else 2
        if stored_v.ndim == single_state_ndim:
            stored_v = stored_v[None, ...]
            stored_w = stored_w[None, ...]
            stored_b = np.asarray([stored_b])
        scores = np.zeros(len(Xtest), dtype=np.float64)
        for member, (V, W, b) in enumerate(zip(stored_v, stored_w, stored_b)):
            model = self._new_latent_model(config, dim, Xtest.shape[1], member)
            model.V, model.W, model.b = (V, W, b)
            scores += model.predict(Xtest)
        return self._write_final(scores / len(stored_v), output)

    def _run_lightgbm(self, config: dict[str, Any], checkpoint: Path) -> dict[str, Any]:
        try:
            import lightgbm as lgb
        except ModuleNotFoundError as exc:
            raise RuntimeError('LightGBM is required: python -m pip install -r requirements.txt') from exc
        matrices = self._lightgbm_matrices(config)
        enc, _ = self._encoded
        ytr, yva = (enc['train'][1], enc['valid'][1])
        uva = enc['valid'][2]
        hp = config['lightgbm_hyperparameters']
        categorical_columns = self._lightgbm_categorical_columns(config)
        # Leave headroom for the isolated worker and evaluator.  Unbounded
        # OpenMP workers were a frequent source of LightGBM worker exits.
        common = dict(learning_rate=hp['learning_rate'], num_leaves=hp['num_leaves'], n_estimators=hp['n_estimators'], min_child_samples=hp['min_child_samples'], subsample=hp['subsample'], colsample_bytree=hp['colsample_bytree'], reg_lambda=hp['reg_lambda'], random_state=config['hyperparameters']['seed'], n_jobs=max(1, min(8, os.cpu_count() or 1)), verbosity=-1)
        callbacks = [lgb.early_stopping(hp['early_stopping_rounds'], verbose=False, first_metric_only=True), lgb.log_evaluation(period=25)]
        started = time.monotonic()
        if config['training_objective'] == 'lambdarank':
            train_order, train_groups = _ranking_order_and_groups(enc['train'][2])
            model = lgb.LGBMRanker(objective='lambdarank', metric='None', lambdarank_truncation_level=6, label_gain=[0, 1], **{**common, 'n_estimators': min(100, hp['n_estimators'])})
            # The official Python evaluator is intentionally run once after
            # fitting. Calling it at every boosting round dominated runtime
            # (minutes of metric bookkeeping for seconds of tree fitting).
            model.fit(matrices['train'][train_order], ytr[train_order], group=train_groups, categorical_feature=categorical_columns, callbacks=[lgb.log_evaluation(period=25)])
            valid_scores = model.predict(matrices['valid'])
        else:
            model = lgb.LGBMClassifier(objective='binary', **common)
            model.fit(matrices['train'], ytr, eval_set=[(matrices['valid'], yva)], eval_metric='binary_logloss', categorical_feature=categorical_columns, callbacks=callbacks)
            valid_scores = model.predict_proba(matrices['valid'])[:, 1]
        valid = self.evaluate_mod.evaluate(uva, yva, valid_scores)
        self._write_validation_slices(checkpoint, valid_scores)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        model.booster_.save_model(str(checkpoint.with_suffix('.txt')))
        best_iteration = int(model.best_iteration_ or model.n_estimators_)
        np.savez_compressed(checkpoint, best_iteration=np.asarray(best_iteration))
        return {**self._metrics(valid), 'best_iteration': best_iteration, 'runtime_seconds': float(time.monotonic() - started)}

    def _finalize_lightgbm(self, config: dict[str, Any], checkpoint: Path, output: Path) -> dict[str, Any]:
        import lightgbm as lgb
        model = lgb.Booster(model_file=str(checkpoint.with_suffix('.txt')))
        return self._write_final(model.predict(self._lightgbm_matrices(config)['test']), output)

    def _sequences(self) -> SequenceContext:
        if self._sequence_context is None:
            cache = self.cache_dir / f'sequence_context_{CACHE_SCHEMA_VERSION}_{self._dataset_fingerprint}.pkl'
            if cache.is_file():
                with cache.open('rb') as handle:
                    self._sequence_context = pickle.load(handle)
            else:
                self._sequence_context = build_previous_positive_context(self._splits, self._encoded[0])
                temporary = cache.with_suffix(f'.{os.getpid()}.tmp')
                with temporary.open('wb') as handle:
                    pickle.dump(self._sequence_context, handle, protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(temporary, cache)
        return self._sequence_context

    def _new_fpmc(self, config: dict[str, Any], context: SequenceContext) -> FPMC:
        hp = config['hyperparameters']
        return FPMC(context.user_count, context.item_count, embedding_dim=hp['embedding_dim'], learning_rate=hp['learning_rate'], l2=hp['l2'], seed=hp['seed'])

    def _new_seq_fm(self, config: dict[str, Any], context: SequenceContext, categorical_dim: int) -> SequentialFM:
        hp = config['hyperparameters']
        return SequentialFM(categorical_dim, context.item_count, embedding_dim=hp['embedding_dim'], learning_rate=hp['learning_rate'], l2=hp['l2'], seed=hp['seed'])

    def _run_seq_fm(self, config: dict[str, Any], checkpoint: Path) -> dict[str, Any]:
        context = self._sequences()
        enc, categorical_dim = self._encoded
        Xtr, ytr, users = enc['train']
        Xva, yva, valid_users = enc['valid']
        hp = config['hyperparameters']
        model = self._new_seq_fm(config, context, categorical_dim)
        rng = np.random.default_rng(hp['seed'])
        best_score, best_state, bad, best_epoch = (-1.0, None, 0, 0)
        started = time.monotonic()
        for epoch in range(1, hp['epochs'] + 1):
            positive, negative = self._bpr_pairs(config, Xtr, users, ytr, rng)
            losses = []
            for start in range(0, len(positive), hp['batch_size']):
                sl = slice(start, start + hp['batch_size'])
                p, n = (positive[sl], negative[sl])
                losses.append(model.bpr_step(Xtr[p], Xtr[n], context.items['train'][p], context.items['train'][n], context.previous_items['train'][p]))
            valid_scores = model.predict(Xva, context.items['valid'], context.previous_items['valid'])
            metrics = self.evaluate_mod.evaluate(valid_users, yva, valid_scores)
            current = self._validation_score(metrics, config)
            print(f"    epoch {epoch:02d} | bpr_loss={np.mean(losses):.6f} | selection={config['hyperparameters'].get('validation_metric', 'primary')}:{current:.6f} | primary={float(metrics['primary']):.6f} | nDCG@5={float(metrics['nDCG@5']):.6f} | best={max(best_score, current):.6f}", flush=True)
            if current > best_score + 1e-05:
                best_score, bad, best_epoch = (current, 0, epoch)
                best_state = {name: getattr(model, name).copy() for name in model.STATE_KEYS}
            else:
                bad += 1
                if bad >= hp['patience']:
                    break
        if best_state is None:
            raise RuntimeError('SequentialFM training produced no checkpoint')
        for name, value in best_state.items():
            setattr(model, name, value)
        model.save(checkpoint, best_epoch=best_epoch)
        valid_scores = model.predict(Xva, context.items['valid'], context.previous_items['valid'])
        valid = self.evaluate_mod.evaluate(valid_users, yva, valid_scores)
        self._write_validation_slices(checkpoint, valid_scores)
        return {'GAUC': float(valid['GAUC']), 'nDCG@5': float(valid['nDCG@5']), 'primary': float(valid['primary']), 'best_epoch': int(best_epoch), 'ensemble_size': 1, 'runtime_seconds': float(time.monotonic() - started)}

    def _run_fpmc(self, config: dict[str, Any], checkpoint: Path) -> dict[str, Any]:
        context = self._sequences()
        enc, _ = self._encoded
        Xtr, ytr, users = enc['train']
        _, yva, valid_users = enc['valid']
        hp = config['hyperparameters']
        model = self._new_fpmc(config, context)
        rng = np.random.default_rng(hp['seed'])
        best_score, best_state, bad, best_epoch = (-1.0, None, 0, 0)
        started = time.monotonic()
        for epoch in range(1, hp['epochs'] + 1):
            positive, negative = self._bpr_pairs(config, Xtr, users, ytr, rng)
            losses = []
            for start in range(0, len(positive), hp['batch_size']):
                sl = slice(start, start + hp['batch_size'])
                p, n = (positive[sl], negative[sl])
                losses.append(model.bpr_step(context.users['train'][p], context.items['train'][p], context.items['train'][n], context.previous_items['train'][p]))
            valid_scores = model.predict(context.users['valid'], context.items['valid'], context.previous_items['valid'])
            metrics = self.evaluate_mod.evaluate(valid_users, yva, valid_scores)
            current = self._validation_score(metrics, config)
            print(f"    epoch {epoch:02d} | bpr_loss={np.mean(losses):.6f} | selection={config['hyperparameters'].get('validation_metric', 'primary')}:{current:.6f} | primary={float(metrics['primary']):.6f} | nDCG@5={float(metrics['nDCG@5']):.6f} | best={max(best_score, current):.6f}", flush=True)
            if current > best_score + 1e-05:
                best_score, bad, best_epoch = (current, 0, epoch)
                best_state = {name: getattr(model, name).copy() for name in model.STATE_KEYS}
            else:
                bad += 1
                if bad >= hp['patience']:
                    break
        if best_state is None:
            raise RuntimeError('FPMC training produced no checkpoint')
        for name, value in best_state.items():
            setattr(model, name, value)
        model.save(checkpoint, best_epoch=best_epoch)
        valid_scores = model.predict(context.users['valid'], context.items['valid'], context.previous_items['valid'])
        valid = self.evaluate_mod.evaluate(valid_users, yva, valid_scores)
        self._write_validation_slices(checkpoint, valid_scores)
        return {'GAUC': float(valid['GAUC']), 'nDCG@5': float(valid['nDCG@5']), 'primary': float(valid['primary']), 'best_epoch': int(best_epoch), 'ensemble_size': 1, 'runtime_seconds': float(time.monotonic() - started)}

    def _finalize_fpmc(self, config: dict[str, Any], checkpoint: Path, output: Path) -> dict[str, Any]:
        context = self._sequences()
        model = self._new_fpmc(config, context)
        model.load(checkpoint)
        scores = model.predict(context.users['test'], context.items['test'], context.previous_items['test'])
        return self._write_final(scores, output)

    def _finalize_seq_fm(self, config: dict[str, Any], checkpoint: Path, output: Path) -> dict[str, Any]:
        context = self._sequences()
        enc, categorical_dim = self._encoded
        model = self._new_seq_fm(config, context, categorical_dim)
        model.load(checkpoint)
        scores = model.predict(enc['test'][0], context.items['test'], context.previous_items['test'])
        return self._write_final(scores, output)

    def _lightgcn_graph(self) -> tuple[SequenceContext, np.ndarray, np.ndarray]:
        context = self._sequences()
        labels = np.asarray(self._encoded[0]['train'][1])
        positive = np.flatnonzero(labels > 0.5)
        edges = np.unique(np.column_stack((
            context.users['train'][positive], context.items['train'][positive]
        )).astype(np.int64), axis=0)
        return context, edges[:, 0], edges[:, 1]

    def _new_lightgcn(self, config: dict[str, Any]) -> LightGCN:
        context, edge_users, edge_items = self._lightgcn_graph()
        hp = config['hyperparameters']
        return LightGCN(
            context.user_count, context.item_count,
            torch.as_tensor(edge_users), torch.as_tensor(edge_items),
            embedding_dim=int(hp['embedding_dim']), num_layers=int(hp['graph_layers']),
        )

    @staticmethod
    def _predict_lightgcn(
        model: LightGCN, users: np.ndarray, items: np.ndarray, *,
        batch_size: int, device: str,
    ) -> np.ndarray:
        model.eval()
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            propagated = model.propagate()
            for start in range(0, len(users), batch_size):
                sl = slice(start, start + batch_size)
                user = torch.as_tensor(users[sl], dtype=torch.long, device=device)
                item = torch.as_tensor(items[sl], dtype=torch.long, device=device)
                outputs.append(model.score(user, item, propagated).cpu().numpy())
        scores = np.concatenate(outputs)
        if not np.all(np.isfinite(scores)):
            raise RuntimeError('lightgcn produced non-finite predictions')
        return scores

    def _run_lightgcn(self, config: dict[str, Any], checkpoint: Path) -> dict[str, Any]:
        context, edge_users, edge_items = self._lightgcn_graph()
        _, yva, valid_users = self._encoded[0]['valid']
        hp = config['hyperparameters']
        seed = int(hp['seed'])
        device = self._device()
        self._set_torch_seed(seed)
        model = self._new_lightgcn(config).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(hp['learning_rate']), weight_decay=float(hp['l2'])
        )
        rng = np.random.default_rng(seed)
        best_score, best_state, best_epoch, bad = (-1.0, None, 0, 0)
        started = time.monotonic()
        for epoch in range(1, int(hp['epochs']) + 1):
            model.train()
            order = rng.permutation(len(edge_users))
            # Full-graph propagation is the expensive operation; one graph BPR
            # update per epoch preserves LightGCN semantics and avoids repeating
            # it for every minibatch.
            if len(order) > 500_000:
                order = order[:500_000]
            users_np = edge_users[order]
            positives_np = edge_items[order]
            negatives_np = rng.integers(0, context.item_count, size=len(order), dtype=np.int64)
            collision = negatives_np == positives_np
            while collision.any():
                negatives_np[collision] = rng.integers(0, context.item_count, size=int(collision.sum()))
                collision = negatives_np == positives_np
            users_t = torch.as_tensor(users_np, dtype=torch.long, device=device)
            positives_t = torch.as_tensor(positives_np, dtype=torch.long, device=device)
            negatives_t = torch.as_tensor(negatives_np, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            propagated = model.propagate()
            positive_score = model.score(users_t, positives_t, propagated)
            negative_score = model.score(users_t, negatives_t, propagated)
            loss = -torch.nn.functional.logsigmoid(positive_score - negative_score).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scores = self._predict_lightgcn(
                model, context.users['valid'], context.items['valid'],
                batch_size=int(hp['batch_size']), device=device,
            )
            metrics = self.evaluate_mod.evaluate(valid_users, yva, scores)
            current = self._validation_score(metrics, config)
            print(f"    lightgcn epoch {epoch:02d} | loss={float(loss.detach()):.6f} | selection={config['hyperparameters'].get('validation_metric', 'primary')}:{current:.6f} | primary={float(metrics['primary']):.6f} | nDCG@5={float(metrics['nDCG@5']):.6f} | best={max(best_score, current):.6f}", flush=True)
            if current > best_score + 1e-5:
                best_score, best_epoch, bad = current, epoch, 0
                best_state = self._cpu_state(model)
            else:
                bad += 1
                if bad >= int(hp['patience']):
                    break
        if best_state is None:
            raise RuntimeError('lightgcn training produced no checkpoint')
        model.load_state_dict(best_state)
        scores = self._predict_lightgcn(
            model, context.users['valid'], context.items['valid'],
            batch_size=int(hp['batch_size']), device=device,
        )
        valid = self.evaluate_mod.evaluate(valid_users, yva, scores)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        self._write_validation_slices(checkpoint, scores)
        torch.save({
            'state_dict': best_state, 'best_epoch': best_epoch,
            'user_count': context.user_count, 'item_count': context.item_count,
            'graph_layers': int(hp['graph_layers']),
        }, checkpoint)
        return {**self._metrics(valid), 'best_epoch': best_epoch,
                'runtime_seconds': float(time.monotonic() - started)}

    def _load_lightgcn_predictions(
        self, config: dict[str, Any], saved: dict[str, Any], split: str,
    ) -> np.ndarray:
        context = self._sequences()
        if (int(saved['user_count']) != context.user_count or
                int(saved['item_count']) != context.item_count):
            raise RuntimeError('lightgcn graph vocabulary mismatch')
        device = self._device()
        model = self._new_lightgcn(config).to(device)
        model.load_state_dict(saved['state_dict'])
        return self._predict_lightgcn(
            model, context.users[split], context.items[split],
            batch_size=int(config['hyperparameters']['batch_size']), device=device,
        )

    def _finalize_lightgcn(self, config: dict[str, Any], checkpoint: Path, output: Path) -> dict[str, Any]:
        return self._write_final(
            self._load_lightgcn_predictions(config, self._torch_load(checkpoint), 'test'), output
        )

    def _run_lightgcn_hybrid(self, config: dict[str, Any], checkpoint: Path) -> dict[str, Any]:
        started = time.monotonic()
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        graph_path = checkpoint.with_suffix('.graph.pt')
        fm_path = checkpoint.with_suffix('.fm.npz')
        graph_config = copy.deepcopy(config)
        graph_config['model'] = 'lightgcn'
        fm_config = copy.deepcopy(config)
        fm_config['model'] = 'fm'
        fm_config['training_objective'] = 'bpr'
        self._run_lightgcn(graph_config, graph_path)
        self._run_latent(fm_config, fm_path)
        graph_state = self._torch_load(graph_path)
        graph_scores = self._load_lightgcn_predictions(graph_config, graph_state, 'valid')
        fm_enc, dim = self._encoded_for(fm_config)
        with np.load(fm_path) as state:
            fm_state = {name: state[name] for name in ('V', 'W', 'b')}
        fm = self._new_latent_model(fm_config, dim, fm_enc['valid'][0].shape[1], int(config['hyperparameters']['seed']))
        fm.V, fm.W, fm.b = fm_state['V'], fm_state['W'], fm_state['b']
        fm_scores = fm.predict(fm_enc['valid'][0])
        users = self._encoded[0]['valid'][2]
        labels = self._encoded[0]['valid'][1]
        graph_rank = self._blend_component(users, graph_scores, config)
        fm_rank = self._blend_component(users, fm_scores, config)
        best = None
        for graph_weight in np.linspace(0.0, 1.0, 11):
            scores = graph_weight * graph_rank + (1.0 - graph_weight) * fm_rank
            metrics = self.evaluate_mod.evaluate(users, labels, scores)
            selected = self._validation_score(metrics, config)
            if best is None or selected > best[0]:
                best = (selected, float(graph_weight), metrics, scores)
        torch.save({'graph': graph_state, 'fm': fm_state, 'graph_weight': best[1]}, checkpoint)
        self._write_validation_slices(checkpoint, best[3])
        for path in (graph_path, graph_path.with_suffix('.slices.json'), fm_path, fm_path.with_suffix('.slices.json')):
            path.unlink(missing_ok=True)
        return {**self._metrics(best[2]), 'graph_weight': best[1],
                'runtime_seconds': float(time.monotonic() - started)}

    def _finalize_lightgcn_hybrid(self, config: dict[str, Any], checkpoint: Path, output: Path) -> dict[str, Any]:
        saved = self._torch_load(checkpoint)
        graph_config = copy.deepcopy(config)
        graph_config['model'] = 'lightgcn'
        graph_scores = self._load_lightgcn_predictions(graph_config, saved['graph'], 'test')
        fm_config = copy.deepcopy(config)
        fm_config['model'] = 'fm'
        fm_config['training_objective'] = 'bpr'
        fm_enc, dim = self._encoded_for(fm_config)
        fm = self._new_latent_model(fm_config, dim, fm_enc['test'][0].shape[1], int(config['hyperparameters']['seed']))
        fm.V, fm.W, fm.b = saved['fm']['V'], saved['fm']['W'], saved['fm']['b']
        users = self._encoded[0]['test'][2]
        weight = float(saved['graph_weight'])
        scores = (weight * self._blend_component(users, graph_scores, config) +
                  (1.0 - weight) * self._blend_component(users, fm.predict(fm_enc['test'][0]), config))
        return self._write_final(scores, output)

    @staticmethod
    def _device() -> str:
        return 'cuda' if torch.cuda.is_available() else 'cpu'

    @staticmethod
    def _set_torch_seed(seed: int) -> None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _cpu_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
        return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    @staticmethod
    def _torch_load(checkpoint: Path):
        try:
            return torch.load(checkpoint, map_location='cpu', weights_only=False)
        except TypeError:
            return torch.load(checkpoint, map_location='cpu')

    def _localize_encoded(self, enc):
        Xtr = np.asarray(enc['train'][0], dtype=np.int64)
        vocabularies = [np.unique(Xtr[:, col]) for col in range(Xtr.shape[1])]
        field_dims = [len(vocab) + 1 for vocab in vocabularies]

        def transform(X):
            X = np.asarray(X, dtype=np.int64)
            out = np.empty_like(X, dtype=np.int64)
            for col, vocab in enumerate(vocabularies):
                values = X[:, col]
                pos = np.searchsorted(vocab, values)
                clipped = np.minimum(pos, len(vocab) - 1)
                known = (pos < len(vocab)) & (vocab[clipped] == values)
                out[:, col] = np.where(known, pos, len(vocab))
            return out
        return ({split: (transform(X), y, users) for split, (X, y, users) in enc.items()}, field_dims)

    def _neural_data(self, config: dict[str, Any], *, base_only: bool=False):
        enc = self._encoded if base_only else self._encoded_for(config)
        if isinstance(enc, tuple):
            enc = enc[0]
        return self._localize_encoded(enc)

    def _dense_neural_data(self, config: dict[str, Any]):
        enc, field_dims = self._neural_data(config)
        columns: dict[str, list[np.ndarray]] = {split: [] for split in self._splits}
        for split, rows in self._splits.items():
            columns[split].append(np.log1p(np.asarray([row[5] for row in rows], dtype=np.float32)))
            if config['features'].get('hour'):
                hour = np.asarray([row[9] for row in rows], dtype=np.float32)
                columns[split].extend((np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)))
            if config['features'].get('weekday'):
                weekday = np.asarray([row[10] for row in rows], dtype=np.float32)
                columns[split].extend((np.sin(2 * np.pi * weekday / 7), np.cos(2 * np.pi * weekday / 7)))
        for feature, enabled in config['features'].items():
            if not enabled or feature in RAW_CATEGORICAL_FEATURES or feature in {DURATION_FEATURE, 'continuous_history_stats', 'hour', 'weekday'}:
                continue
            values, _ = self._feature_values(feature, categorical=False)
            for split in self._splits:
                columns[split].append(values[split].astype(np.float32))
        numeric = {split: np.column_stack(values).astype(np.float32) for split, values in columns.items()}
        mean = numeric['train'].mean(axis=0, keepdims=True)
        scale = numeric['train'].std(axis=0, keepdims=True)
        scale[scale < 1e-6] = 1.0
        numeric = {split: ((values - mean) / scale).astype(np.float32) for split, values in numeric.items()}
        return enc, field_dims, numeric

    @staticmethod
    def _predict_static(model: torch.nn.Module, X: np.ndarray, *, batch_size: int, device: str, numeric: np.ndarray | None=None) -> np.ndarray:
        model.eval()
        outputs = []
        with torch.no_grad():
            for start in range(0, len(X), batch_size):
                batch = torch.as_tensor(X[start:start + batch_size], dtype=torch.long, device=device)
                dense = None if numeric is None else torch.as_tensor(numeric[start:start + batch_size], dtype=torch.float32, device=device)
                outputs.append(model(batch, dense).detach().cpu().numpy() if dense is not None else model(batch).detach().cpu().numpy())
        scores = np.concatenate(outputs)
        if not np.all(np.isfinite(scores)):
            raise RuntimeError('model produced non-finite predictions')
        return scores

    @staticmethod
    def _predict_multitask(model: MultiTaskRecommender, X: np.ndarray, *, batch_size: int, device: str) -> np.ndarray:
        model.eval()
        outputs = []
        with torch.no_grad():
            for start in range(0, len(X), batch_size):
                batch = torch.as_tensor(X[start:start + batch_size], dtype=torch.long, device=device)
                outputs.append(model(batch)['long_view'].detach().cpu().numpy())
        scores = np.concatenate(outputs)
        if not np.all(np.isfinite(scores)):
            raise RuntimeError('multitask model produced non-finite predictions')
        return scores

    def _new_static_neural(self, model_name: str, field_dims: list[int], config: dict[str, Any], numeric_dim: int=0) -> torch.nn.Module:
        hp = config['hyperparameters']
        common = dict(field_dims=field_dims, embedding_dim=int(hp['embedding_dim']), hidden_dims=(128, 64), dropout=float(hp['dropout']))
        if model_name == 'deepfm':
            return DeepFM(**common)
        if model_name == 'dcnv2':
            return DCNv2(cross_layers=3, **common)
        if model_name == 'dcnv2_dense':
            return DCNv2(cross_layers=3, numeric_dim=numeric_dim, **common)
        if model_name == 'two_tower':
            return TwoTowerCandidateModel(
                field_dims=field_dims, embedding_dim=int(hp['embedding_dim']),
                numeric_dim=numeric_dim, tower_dim=64, dropout=float(hp['dropout']),
            )
        raise ValueError(f'unsupported static neural model: {model_name}')

    def _mine_static_negatives(
        self, model: torch.nn.Module, X: np.ndarray, numeric: np.ndarray | None,
        positive: np.ndarray, pool: np.ndarray, keep: int, *, batch_size: int,
        device: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        selected = np.empty((len(positive), keep), dtype=np.int64)
        was_training = model.training
        model.eval()
        group_batch = max(1, batch_size // pool.shape[1])
        with torch.no_grad():
            for start in range(0, len(positive), group_batch):
                candidates = pool[start:start + group_batch]
                flat = candidates.reshape(-1)
                bx = torch.as_tensor(X[flat], dtype=torch.long, device=device)
                dense = None if numeric is None else torch.as_tensor(
                    numeric[flat], dtype=torch.float32, device=device
                )
                score = model(bx, dense) if dense is not None else model(bx)
                score = score.reshape(len(candidates), candidates.shape[1])
                top = torch.topk(score, k=keep, dim=1).indices.cpu().numpy()
                selected[start:start + len(candidates)] = np.take_along_axis(candidates, top, axis=1)
        model.train(was_training)
        return np.repeat(positive, keep), selected.reshape(-1)

    def _run_static_neural(self, config: dict[str, Any], checkpoint: Path, model_name: str) -> dict[str, Any]:
        dense_model = model_name in {'dcnv2_dense', 'two_tower'}
        if dense_model:
            enc, field_dims, numeric = self._dense_neural_data(config)
        else:
            enc, field_dims = self._neural_data(config)
            numeric = None
        Xtr, ytr, users = enc['train']
        Xva, yva, valid_users = enc['valid']
        original_Xtr = self._encoded_for(config)[0]['train'][0]
        hp = config['hyperparameters']
        seed = int(hp['seed'])
        batch_size = int(hp['batch_size'])
        device = self._device()
        self._set_torch_seed(seed)
        numeric_dim = 0 if numeric is None else int(numeric['train'].shape[1])
        model = self._new_static_neural(model_name, field_dims, config, numeric_dim).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(hp['learning_rate']), weight_decay=float(hp['l2']))
        rng = np.random.default_rng(seed)
        best_score, best_state, best_epoch, bad = (-1.0, None, 0, 0)
        started = time.monotonic()
        for epoch in range(1, int(hp['epochs']) + 1):
            model.train()
            losses = []
            if config['training_objective'] == 'bpr':
                hard_pool = int(hp['hard_negative_pool_size'])
                if hard_pool:
                    strategy = hp['negative_sampling_strategy']
                    match = {'random': None, 'same_tab': original_Xtr[:, 3],
                             'same_author': original_Xtr[:, 2]}[strategy]
                    base_positive, pool = build_group_softmax_indices(
                        users, ytr, rng, max(hard_pool, int(hp['negatives_per_positive'])), match
                    )
                    if len(base_positive) > 100_000:
                        base_positive, pool = base_positive[:100_000], pool[:100_000]
                    positive, negative = self._mine_static_negatives(
                        model, Xtr, None if numeric is None else numeric['train'],
                        base_positive, pool, int(hp['negatives_per_positive']),
                        batch_size=batch_size, device=device,
                    )
                else:
                    positive, negative = self._bpr_pairs(config, original_Xtr, users, ytr, rng)
                for start in range(0, len(positive), batch_size):
                    sl = slice(start, start + batch_size)
                    p, n = (positive[sl], negative[sl])
                    pos_x = torch.as_tensor(Xtr[p], dtype=torch.long, device=device)
                    neg_x = torch.as_tensor(Xtr[n], dtype=torch.long, device=device)
                    pos_numeric = None if numeric is None else torch.as_tensor(numeric['train'][p], dtype=torch.float32, device=device)
                    neg_numeric = None if numeric is None else torch.as_tensor(numeric['train'][n], dtype=torch.float32, device=device)
                    optimizer.zero_grad(set_to_none=True)
                    pos_scores = model(pos_x, pos_numeric) if pos_numeric is not None else model(pos_x)
                    neg_scores = model(neg_x, neg_numeric) if neg_numeric is not None else model(neg_x)
                    loss = -torch.nn.functional.logsigmoid(pos_scores - neg_scores).mean()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                    losses.append(float(loss.detach().cpu()))
            elif config['training_objective'] == 'group_softmax':
                hp_strategy = hp['negative_sampling_strategy']
                match_values = {'random': None, 'same_tab': original_Xtr[:, 3], 'same_author': original_Xtr[:, 2]}[hp_strategy]
                keep = int(hp['negatives_per_positive'])
                hard_pool = int(hp['hard_negative_pool_size'])
                positive, negative = build_group_softmax_indices(
                    users, ytr, rng, max(keep, hard_pool), match_values
                )
                if hard_pool:
                    if len(positive) > 100_000:
                        positive, negative = positive[:100_000], negative[:100_000]
                    repeated, mined = self._mine_static_negatives(
                        model, Xtr, None if numeric is None else numeric['train'],
                        positive, negative, keep, batch_size=batch_size, device=device,
                    )
                    positive = repeated.reshape(-1, keep)[:, 0]
                    negative = mined.reshape(-1, keep)
                for start in range(0, len(positive), batch_size):
                    p = positive[start:start + batch_size]
                    n = negative[start:start + batch_size]
                    pos_x = torch.as_tensor(Xtr[p], dtype=torch.long, device=device)
                    neg_x = torch.as_tensor(Xtr[n.reshape(-1)], dtype=torch.long, device=device)
                    pos_numeric = None if numeric is None else torch.as_tensor(numeric['train'][p], dtype=torch.float32, device=device)
                    neg_numeric = None if numeric is None else torch.as_tensor(numeric['train'][n.reshape(-1)], dtype=torch.float32, device=device)
                    optimizer.zero_grad(set_to_none=True)
                    pos_score = model(pos_x, pos_numeric) if pos_numeric is not None else model(pos_x)
                    neg_score = model(neg_x, neg_numeric) if neg_numeric is not None else model(neg_x)
                    logits = torch.cat((pos_score[:, None], neg_score.reshape(len(p), -1)), dim=1)
                    loss = torch.nn.functional.cross_entropy(
                        logits, torch.zeros(len(p), dtype=torch.long, device=device)
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                    losses.append(float(loss.detach().cpu()))
            elif config['training_objective'] == 'bce':
                indices = rng.permutation(len(ytr))
                for start in range(0, len(indices), batch_size):
                    batch = indices[start:start + batch_size]
                    bx = torch.as_tensor(Xtr[batch], dtype=torch.long, device=device)
                    by = torch.as_tensor(ytr[batch], dtype=torch.float32, device=device)
                    dense = None if numeric is None else torch.as_tensor(numeric['train'][batch], dtype=torch.float32, device=device)
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(bx, dense) if dense is not None else model(bx)
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, by)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                    losses.append(float(loss.detach().cpu()))
            else:
                raise ValueError(f"{model_name} supports only bce/bpr/group_softmax, got {config['training_objective']!r}")
            scores = self._predict_static(model, Xva, batch_size=batch_size, device=device, numeric=None if numeric is None else numeric['valid'])
            metrics = self.evaluate_mod.evaluate(valid_users, yva, scores)
            current = self._validation_score(metrics, config)
            print(f"    {model_name} epoch {epoch:02d} | loss={np.mean(losses):.6f} | selection={config['hyperparameters'].get('validation_metric', 'primary')}:{current:.6f} | primary={float(metrics['primary']):.6f} | nDCG@5={float(metrics['nDCG@5']):.6f} | best={max(best_score, current):.6f}", flush=True)
            if current > best_score + 1e-05:
                best_score, best_epoch, bad = (current, epoch, 0)
                best_state = self._cpu_state(model)
            else:
                bad += 1
                if bad >= int(hp['patience']):
                    break
        if best_state is None:
            raise RuntimeError(f'{model_name} training produced no checkpoint')
        model.load_state_dict(best_state)
        valid_scores = self._predict_static(model, Xva, batch_size=batch_size, device=device, numeric=None if numeric is None else numeric['valid'])
        valid = self.evaluate_mod.evaluate(valid_users, yva, valid_scores)
        self._write_validation_slices(checkpoint, valid_scores)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'model_name': model_name, 'field_dims': field_dims, 'numeric_dim': numeric_dim, 'state_dict': best_state, 'best_epoch': best_epoch}, checkpoint)
        return {'GAUC': float(valid['GAUC']), 'nDCG@5': float(valid['nDCG@5']), 'primary': float(valid['primary']), 'best_epoch': int(best_epoch), 'ensemble_size': 1, 'runtime_seconds': float(time.monotonic() - started)}

    def _run_deepfm(self, config: dict[str, Any], checkpoint: Path) -> dict[str, Any]:
        return self._run_static_neural(config, checkpoint, 'deepfm')

    def _run_dcnv2(self, config: dict[str, Any], checkpoint: Path) -> dict[str, Any]:
        return self._run_static_neural(config, checkpoint, 'dcnv2')

    def _run_dcnv2_dense(self, config: dict[str, Any], checkpoint: Path) -> dict[str, Any]:
        return self._run_static_neural(config, checkpoint, 'dcnv2_dense')

    def _run_two_tower(self, config: dict[str, Any], checkpoint: Path) -> dict[str, Any]:
        return self._run_static_neural(config, checkpoint, 'two_tower')

    def _finalize_static_neural(self, config: dict[str, Any], checkpoint: Path, output: Path, model_name: str) -> dict[str, Any]:
        if model_name in {'dcnv2_dense', 'two_tower'}:
            enc, field_dims, numeric = self._dense_neural_data(config)
        else:
            enc, field_dims = self._neural_data(config)
            numeric = None
        saved = self._torch_load(checkpoint)
        if [int(v) for v in saved['field_dims']] != field_dims:
            raise RuntimeError(f'{model_name} categorical vocabulary mismatch')
        device = self._device()
        numeric_dim = 0 if numeric is None else int(numeric['test'].shape[1])
        if int(saved.get('numeric_dim', 0)) != numeric_dim:
            raise RuntimeError(f'{model_name} dense feature width mismatch')
        model = self._new_static_neural(model_name, field_dims, config, numeric_dim).to(device)
        model.load_state_dict(saved['state_dict'])
        scores = self._predict_static(model, enc['test'][0], batch_size=int(config['hyperparameters']['batch_size']), device=device, numeric=None if numeric is None else numeric['test'])
        return self._write_final(scores, output)

    def _finalize_deepfm(self, config: dict[str, Any], checkpoint: Path, output: Path) -> dict[str, Any]:
        return self._finalize_static_neural(config, checkpoint, output, 'deepfm')

    def _finalize_dcnv2(self, config: dict[str, Any], checkpoint: Path, output: Path) -> dict[str, Any]:
        return self._finalize_static_neural(config, checkpoint, output, 'dcnv2')

    def _finalize_dcnv2_dense(self, config: dict[str, Any], checkpoint: Path, output: Path) -> dict[str, Any]:
        return self._finalize_static_neural(config, checkpoint, output, 'dcnv2_dense')

    def _finalize_two_tower(self, config: dict[str, Any], checkpoint: Path, output: Path) -> dict[str, Any]:
        return self._finalize_static_neural(config, checkpoint, output, 'two_tower')

    @staticmethod
    def _within_user_rank(users, scores: np.ndarray) -> np.ndarray:
        ranked = np.zeros(len(scores), dtype=np.float32)
        groups: dict[Any, list[int]] = defaultdict(list)
        for index, user in enumerate(users):
            groups[user].append(index)
        for indices in groups.values():
            order = np.argsort(np.asarray(scores)[indices], kind='stable')
            denominator = max(1, len(indices) - 1)
            for position, local_index in enumerate(order):
                ranked[indices[int(local_index)]] = position / denominator
        return ranked

    @staticmethod
    def _within_user_zscore(users, scores: np.ndarray) -> np.ndarray:
        """Normalize each component within a user before heterogeneous blending.

        Rank blending is robust but discards confidence gaps.  Per-user
        z-scores retain those gaps while avoiding incomparable global scales
        between FM, neural, and tree predictors.  Constant groups map to zero,
        which is deterministic and cannot change their ordering.
        """
        normalized = np.zeros(len(scores), dtype=np.float32)
        groups: dict[Any, list[int]] = defaultdict(list)
        values = np.asarray(scores, dtype=np.float64)
        for index, user in enumerate(users):
            groups[user].append(index)
        for indices in groups.values():
            group = values[indices]
            scale = float(group.std())
            if scale <= 1e-12:
                continue
            normalized[indices] = ((group - float(group.mean())) / scale).astype(np.float32)
        return normalized

    def _blend_component(self, users, scores: np.ndarray, config: dict[str, Any]) -> np.ndarray:
        mode = str(config.get("hyperparameters", {}).get("blend_mode", "rank"))
        return (self._within_user_zscore if mode == "zscore" else self._within_user_rank)(users, scores)

    def _component_predictions(self, config: dict[str, Any], dcn_state: dict[str, Any], fm_state: dict[str, np.ndarray], split: str, tree_model: str | None=None):
        dcn_config = copy.deepcopy(config)
        dcn_name = str(dcn_state.get('model_name') or 'dcnv2')
        dcn_config['model'] = dcn_name
        dcn_config['training_objective'] = 'bce'
        if dcn_name == 'dcnv2_dense':
            enc, field_dims, numeric = self._dense_neural_data(dcn_config)
        else:
            enc, field_dims = self._neural_data(dcn_config)
            numeric = None
        device = self._device()
        numeric_dim = 0 if numeric is None else numeric[split].shape[1]
        dcn = self._new_static_neural(dcn_name, field_dims, dcn_config, numeric_dim).to(device)
        dcn.load_state_dict(dcn_state['state_dict'])
        dcn_scores = self._predict_static(
            dcn, enc[split][0], batch_size=int(config['hyperparameters']['batch_size']),
            device=device, numeric=None if numeric is None else numeric[split],
        )
        fm_config = copy.deepcopy(config)
        fm_config['model'] = 'fm'
        fm_config['training_objective'] = 'bpr'
        fm_enc, dim = self._encoded_for(fm_config)
        fm = self._new_latent_model(fm_config, dim, fm_enc[split][0].shape[1], int(config['hyperparameters']['seed']))
        fm.V, fm.W, fm.b = fm_state['V'], fm_state['W'], fm_state['b']
        scores = [dcn_scores, fm.predict(fm_enc[split][0])]
        if tree_model is not None:
            import lightgbm as lgb
            booster = lgb.Booster(model_str=tree_model)
            scores.append(booster.predict(self._lightgbm_matrices(config)[split]))
        return tuple(scores)

    def _run_hybrid_blend(self, config: dict[str, Any], checkpoint: Path) -> dict[str, Any]:
        """Train dense-DCN, FM and tree components, then validation rank-blend."""
        started = time.monotonic()
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        dcn_path = checkpoint.with_suffix('.dcn.pt')
        fm_path = checkpoint.with_suffix('.fm.npz')
        tree_path = checkpoint.with_suffix('.tree.npz')
        dcn_config = copy.deepcopy(config)
        dcn_config['model'] = 'dcnv2'
        dcn_config['training_objective'] = 'bce'
        fm_config = copy.deepcopy(config)
        fm_config['model'] = 'fm'
        fm_config['training_objective'] = 'bpr'
        tree_config = copy.deepcopy(config)
        tree_config['model'] = 'lightgbm'
        tree_config['training_objective'] = 'lambdarank'
        self._run_static_neural(dcn_config, dcn_path, 'dcnv2')
        self._run_latent(fm_config, fm_path)
        self._run_lightgbm(tree_config, tree_path)
        dcn_state = self._torch_load(dcn_path)
        with np.load(fm_path) as state:
            fm_state = {key: state[key] for key in ('V', 'W', 'b')}
        tree_txt = tree_path.with_suffix('.txt')
        import lightgbm as lgb
        tree_model = lgb.Booster(model_file=str(tree_txt)).model_to_string()
        dcn_scores, fm_scores, tree_scores = self._component_predictions(config, dcn_state, fm_state, 'valid', tree_model)
        users = self._encoded[0]['valid'][2]
        labels = self._encoded[0]['valid'][1]
        dcn_rank = self._blend_component(users, dcn_scores, config)
        fm_rank = self._blend_component(users, fm_scores, config)
        tree_rank = self._blend_component(users, tree_scores, config)
        best = None
        for dcn_weight in np.linspace(0.0, 1.0, 11):
            for fm_weight in np.linspace(0.0, 1.0, 11):
                tree_weight = 1.0 - dcn_weight - fm_weight
                if tree_weight < 0 or tree_weight > 1:
                    continue
                scores = dcn_weight * dcn_rank + fm_weight * fm_rank + tree_weight * tree_rank
                metrics = self.evaluate_mod.evaluate(users, labels, scores)
                selected = self._validation_score(metrics, config)
                if best is None or selected > best[0]:
                    best = (selected, (dcn_weight, fm_weight, tree_weight), metrics, scores)
        torch.save({'dcn': dcn_state, 'fm': fm_state, 'tree_model': tree_model, 'weights': best[1]}, checkpoint)
        self._write_validation_slices(checkpoint, best[3])
        dcn_path.unlink(missing_ok=True)
        fm_path.unlink(missing_ok=True)
        tree_path.unlink(missing_ok=True)
        tree_txt.unlink(missing_ok=True)
        return {**self._metrics(best[2]), 'blend_weights': list(best[1]), 'best_epoch': dcn_state.get('best_epoch'), 'runtime_seconds': float(time.monotonic() - started)}

    def _finalize_hybrid_blend(self, config: dict[str, Any], checkpoint: Path, output: Path) -> dict[str, Any]:
        saved = self._torch_load(checkpoint)
        dcn_scores, fm_scores, tree_scores = self._component_predictions(config, saved['dcn'], saved['fm'], 'test', saved['tree_model'])
        users = self._encoded[0]['test'][2]
        dcn_weight, fm_weight, tree_weight = (float(value) for value in saved['weights'])
        scores = (
            dcn_weight * self._blend_component(users, dcn_scores, config)
            + fm_weight * self._blend_component(users, fm_scores, config)
            + tree_weight * self._blend_component(users, tree_scores, config)
        )
        return self._write_final(scores, output)

    def _raw_sidecars(self) -> dict[str, dict[str, np.ndarray]]:
        if self._raw_sidecar_cache is not None:
            return self._raw_sidecar_cache
        store: dict[str, dict[str, list[Any]]] = {split: {'time_ms': [], 'click': [], 'like': []} for split in SPLIT_DATES}
        for filename in STANDARD_LOG_FILES:
            with (self.data_dir / filename).open('r', encoding='utf-8', newline='') as fh:
                for row in csv.DictReader(fh):
                    date = int(row['date'])
                    split = next((name for name, (lo, hi) in SPLIT_DATES.items() if lo <= date <= hi), None)
                    if split is None:
                        continue
                    store[split]['time_ms'].append(int(float(row['time_ms'])))
                    store[split]['click'].append(_binary(row.get('is_click')))
                    store[split]['like'].append(_binary(row.get('is_like')))
        result: dict[str, dict[str, np.ndarray]] = {}
        for split, values in store.items():
            expected = len(self._splits[split])
            if len(values['time_ms']) != expected:
                raise RuntimeError(f"raw sidecar alignment failed for {split}: {len(values['time_ms'])} rows != starter {expected}")
            result[split] = {'time_ms': np.asarray(values['time_ms'], dtype=np.int64), 'click': np.asarray(values['click'], dtype=np.float32), 'like': np.asarray(values['like'], dtype=np.float32)}
        self._raw_sidecar_cache = result
        return result

    def _new_multitask(self, field_dims: list[int], config: dict[str, Any]) -> MultiTaskRecommender:
        hp = config['hyperparameters']
        return MultiTaskRecommender(field_dims=field_dims, embedding_dim=int(hp['embedding_dim']), hidden_dims=(128, 64), tasks=('long_view', 'click', 'like'), dropout=float(hp['dropout']))

    def _run_multitask(self, config: dict[str, Any], checkpoint: Path) -> dict[str, Any]:
        enc, field_dims = self._neural_data(config)
        Xtr, ytr, users = enc['train']
        Xva, yva, valid_users = enc['valid']
        original_Xtr = self._encoded_for(config)[0]['train'][0]
        aux = self._raw_sidecars()['train']
        hp = config['hyperparameters']
        seed = int(hp['seed'])
        batch_size = int(hp['batch_size'])
        device = self._device()
        self._set_torch_seed(seed)
        model = self._new_multitask(field_dims, config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(hp['learning_rate']), weight_decay=float(hp['l2']))
        rng = np.random.default_rng(seed)
        best_score, best_state, best_epoch, bad = (-1.0, None, 0, 0)
        started = time.monotonic()
        auxiliary_weight = float(hp['auxiliary_weight'])

        def aux_bce(outputs, indices):
            loss = torch.zeros((), device=device)
            weights = {'click': auxiliary_weight, 'like': auxiliary_weight * 0.5}
            for task, weight in weights.items():
                target = torch.as_tensor(aux[task][indices], dtype=torch.float32, device=device)
                loss = loss + weight * torch.nn.functional.binary_cross_entropy_with_logits(outputs[task], target)
            return loss

        def mine(positive, pool, keep):
            selected = np.empty((len(positive), keep), dtype=np.int64)
            was_training = model.training
            model.eval()
            group_batch = max(1, batch_size // pool.shape[1])
            with torch.no_grad():
                for begin in range(0, len(positive), group_batch):
                    candidates = pool[begin:begin + group_batch]
                    flat = candidates.reshape(-1)
                    bx = torch.as_tensor(Xtr[flat], dtype=torch.long, device=device)
                    scores = model(bx)['long_view'].reshape(len(candidates), candidates.shape[1])
                    top = torch.topk(scores, k=keep, dim=1).indices.cpu().numpy()
                    selected[begin:begin + len(candidates)] = np.take_along_axis(candidates, top, axis=1)
            model.train(was_training)
            return np.repeat(positive, keep), selected.reshape(-1)

        for epoch in range(1, int(hp['epochs']) + 1):
            model.train()
            losses = []
            if config['training_objective'] == 'bpr':
                hard_pool = int(hp['hard_negative_pool_size'])
                if hard_pool:
                    strategy = hp['negative_sampling_strategy']
                    match = {'random': None, 'same_tab': original_Xtr[:, 3],
                             'same_author': original_Xtr[:, 2]}[strategy]
                    base_positive, pool = build_group_softmax_indices(
                        users, ytr, rng, max(hard_pool, int(hp['negatives_per_positive'])), match
                    )
                    if len(base_positive) > 100_000:
                        base_positive, pool = base_positive[:100_000], pool[:100_000]
                    positive, negative = mine(
                        base_positive, pool, int(hp['negatives_per_positive'])
                    )
                else:
                    positive, negative = self._bpr_pairs(config, original_Xtr, users, ytr, rng)
                for start in range(0, len(positive), batch_size):
                    sl = slice(start, start + batch_size)
                    p, n = (positive[sl], negative[sl])
                    px = torch.as_tensor(Xtr[p], dtype=torch.long, device=device)
                    nx = torch.as_tensor(Xtr[n], dtype=torch.long, device=device)
                    optimizer.zero_grad(set_to_none=True)
                    pout, nout = (model(px), model(nx))
                    main = -torch.nn.functional.logsigmoid(pout['long_view'] - nout['long_view']).mean()
                    loss = main + 0.5 * (aux_bce(pout, p) + aux_bce(nout, n))
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                    losses.append(float(loss.detach().cpu()))
            elif config['training_objective'] == 'group_softmax':
                strategy = hp['negative_sampling_strategy']
                match = {'random': None, 'same_tab': original_Xtr[:, 3],
                         'same_author': original_Xtr[:, 2]}[strategy]
                keep = int(hp['negatives_per_positive'])
                hard_pool = int(hp['hard_negative_pool_size'])
                positive, negative = build_group_softmax_indices(
                    users, ytr, rng, max(keep, hard_pool), match
                )
                if hard_pool:
                    if len(positive) > 100_000:
                        positive, negative = positive[:100_000], negative[:100_000]
                    repeated, flat = mine(positive, negative, keep)
                    positive = repeated.reshape(-1, keep)[:, 0]
                    negative = flat.reshape(-1, keep)
                for start in range(0, len(positive), batch_size):
                    p = positive[start:start + batch_size]
                    n = negative[start:start + batch_size]
                    px = torch.as_tensor(Xtr[p], dtype=torch.long, device=device)
                    nx = torch.as_tensor(Xtr[n.reshape(-1)], dtype=torch.long, device=device)
                    optimizer.zero_grad(set_to_none=True)
                    pout = model(px)
                    nout = model(nx)
                    logits = torch.cat((
                        pout['long_view'][:, None],
                        nout['long_view'].reshape(len(p), n.shape[1]),
                    ), dim=1)
                    main = torch.nn.functional.cross_entropy(
                        logits, torch.zeros(len(p), dtype=torch.long, device=device)
                    )
                    # Auxiliary feedback is used as a target only; it is never
                    # exposed as a current-row input feature.
                    loss = main + aux_bce(pout, p)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                    losses.append(float(loss.detach().cpu()))
            elif config['training_objective'] == 'bce':
                indices = rng.permutation(len(ytr))
                for start in range(0, len(indices), batch_size):
                    batch = indices[start:start + batch_size]
                    bx = torch.as_tensor(Xtr[batch], dtype=torch.long, device=device)
                    by = torch.as_tensor(ytr[batch], dtype=torch.float32, device=device)
                    optimizer.zero_grad(set_to_none=True)
                    outputs = model(bx)
                    main = torch.nn.functional.binary_cross_entropy_with_logits(outputs['long_view'], by)
                    loss = main + aux_bce(outputs, batch)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                    losses.append(float(loss.detach().cpu()))
            else:
                raise ValueError(f"multitask supports only bce/bpr/group_softmax, got {config['training_objective']!r}")
            scores = self._predict_multitask(model, Xva, batch_size=batch_size, device=device)
            metrics = self.evaluate_mod.evaluate(valid_users, yva, scores)
            current = self._validation_score(metrics, config)
            print(f"    multitask epoch {epoch:02d} | loss={np.mean(losses):.6f} | selection={config['hyperparameters'].get('validation_metric', 'primary')}:{current:.6f} | primary={float(metrics['primary']):.6f} | nDCG@5={float(metrics['nDCG@5']):.6f} | best={max(best_score, current):.6f}", flush=True)
            if current > best_score + 1e-05:
                best_score, best_epoch, bad = (current, epoch, 0)
                best_state = self._cpu_state(model)
            else:
                bad += 1
                if bad >= int(hp['patience']):
                    break
        if best_state is None:
            raise RuntimeError('multitask training produced no checkpoint')
        model.load_state_dict(best_state)
        scores = self._predict_multitask(model, Xva, batch_size=batch_size, device=device)
        valid = self.evaluate_mod.evaluate(valid_users, yva, scores)
        self._write_validation_slices(checkpoint, scores)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'field_dims': field_dims, 'state_dict': best_state, 'best_epoch': best_epoch}, checkpoint)
        return {'GAUC': float(valid['GAUC']), 'nDCG@5': float(valid['nDCG@5']), 'primary': float(valid['primary']), 'best_epoch': int(best_epoch), 'ensemble_size': 1, 'runtime_seconds': float(time.monotonic() - started)}

    def _finalize_multitask(self, config: dict[str, Any], checkpoint: Path, output: Path) -> dict[str, Any]:
        enc, field_dims = self._neural_data(config)
        saved = self._torch_load(checkpoint)
        if [int(v) for v in saved['field_dims']] != field_dims:
            raise RuntimeError('multitask categorical vocabulary mismatch')
        device = self._device()
        model = self._new_multitask(field_dims, config).to(device)
        model.load_state_dict(saved['state_dict'])
        scores = self._predict_multitask(model, enc['test'][0], batch_size=int(config['hyperparameters']['batch_size']), device=device)
        return self._write_final(scores, output)

    def _sasrec_data(self, max_seq_len: int=50) -> tuple[dict[str, dict[str, np.ndarray]], int]:
        cached = self._sasrec_cache.get(max_seq_len)
        if cached is not None:
            return cached
        cache_path = self.cache_dir / f'sasrec_{CACHE_SCHEMA_VERSION}_{self._dataset_fingerprint}_{max_seq_len}.pkl'
        if cache_path.is_file():
            with cache_path.open('rb') as handle:
                result = pickle.load(handle)
            self._sasrec_cache[max_seq_len] = result
            return result
        base_local, field_dims = self._neural_data({}, base_only=True)
        sidecars = self._raw_sidecars()
        num_videos = int(field_dims[1])
        candidates = {split: base_local[split][0][:, 1].astype(np.int32) + 1 for split in ('train', 'valid', 'test')}
        train_users = np.asarray(base_local['train'][2], dtype=object)
        train_labels = np.asarray(base_local['train'][1], dtype=np.float32)
        train_times = sidecars['train']['time_ms']
        histories: dict[Any, deque[int]] = defaultdict(lambda: deque(maxlen=max_seq_len))
        train_history = np.zeros((len(train_users), max_seq_len), dtype=np.int32)
        order = np.argsort(train_times, kind='stable')
        cursor = 0
        while cursor < len(order):
            timestamp = train_times[order[cursor]]
            end = cursor + 1
            while end < len(order) and train_times[order[end]] == timestamp:
                end += 1
            same_time = order[cursor:end]
            for idx in same_time:
                seq = list(histories[train_users[idx]])
                if seq:
                    train_history[idx, :len(seq)] = seq
            for idx in same_time:
                if train_labels[idx] > 0.5:
                    histories[train_users[idx]].append(int(candidates['train'][idx]))
            cursor = end

        def safe_mask(history: np.ndarray) -> np.ndarray:
            mask = history != 0
            empty = ~mask.any(axis=1)
            mask[empty, 0] = True
            return mask
        data = {'train': {'history': train_history, 'mask': safe_mask(train_history), 'candidate': candidates['train']}}
        for split in ('valid', 'test'):
            users = np.asarray(base_local[split][2], dtype=object)
            history = np.zeros((len(users), max_seq_len), dtype=np.int32)
            for idx, user in enumerate(users):
                seq = list(histories[user])
                if seq:
                    history[idx, :len(seq)] = seq
            data[split] = {'history': history, 'mask': safe_mask(history), 'candidate': candidates[split]}
        result = (data, num_videos)
        temporary = cache_path.with_suffix(f'.{os.getpid()}.tmp')
        with temporary.open('wb') as handle:
            pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, cache_path)
        self._sasrec_cache[max_seq_len] = result
        return result

    def _metadata_history_data(self, max_seq_len: int=20):
        cached = self._metadata_history_cache.get(max_seq_len)
        if cached is not None:
            return cached
        cache_path = self.cache_dir / f'metadata_history_{CACHE_SCHEMA_VERSION}_{self._dataset_fingerprint}_{max_seq_len}.pkl'
        if cache_path.is_file():
            with cache_path.open('rb') as handle:
                result = pickle.load(handle)
            self._metadata_history_cache[max_seq_len] = result
            return result
        base, field_dims = self._neural_data({}, base_only=True)
        tags, tag_width = self._raw_categorical('tag')
        metadata_dims = {
            'item': int(field_dims[1]), 'author': int(field_dims[2]),
            'tag': int(tag_width), 'duration': int(field_dims[4]),
        }
        candidate: dict[str, dict[str, np.ndarray]] = {}
        for split in self._splits:
            X = base[split][0]
            candidate[split] = {
                'item': X[:, 1].astype(np.int32) + 1,
                'author': X[:, 2].astype(np.int32) + 1,
                'tag': tags[split].astype(np.int32) + 1,
                'duration': X[:, 4].astype(np.int32) + 1,
            }
        histories: dict[Any, deque[tuple[int, int, int, int]]] = defaultdict(
            lambda: deque(maxlen=max_seq_len)
        )

        def allocate(length: int) -> dict[str, np.ndarray]:
            return {
                name: np.zeros((length, max_seq_len), dtype=np.int32)
                for name in metadata_dims
            }

        train_rows = self._splits['train']
        train_history = allocate(len(train_rows))
        order = sorted(range(len(train_rows)), key=lambda index: (int(train_rows[index][8]), index))
        start = 0
        names = tuple(metadata_dims)
        while start < len(order):
            timestamp = int(train_rows[order[start]][8])
            end = start + 1
            while end < len(order) and int(train_rows[order[end]][8]) == timestamp:
                end += 1
            for index in order[start:end]:
                sequence = list(histories[train_rows[index][1]])
                for position, token in enumerate(sequence):
                    for field, value in zip(names, token):
                        train_history[field][index, position] = value
            for index in order[start:end]:
                if int(train_rows[index][6]) == 1:
                    histories[train_rows[index][1]].append(tuple(
                        int(candidate['train'][field][index]) for field in names
                    ))
            start = end
        data = {'train': {**train_history, **{f'candidate_{name}': values for name, values in candidate['train'].items()}}}
        data['train']['mask'] = train_history['item'] != 0
        for split in ('valid', 'test'):
            rows = self._splits[split]
            history = allocate(len(rows))
            for index, row in enumerate(rows):
                for position, token in enumerate(histories[row[1]]):
                    for field, value in zip(names, token):
                        history[field][index, position] = value
            data[split] = {**history, **{f'candidate_{name}': values for name, values in candidate[split].items()}}
            data[split]['mask'] = history['item'] != 0
        result = (data, metadata_dims)
        temporary = cache_path.with_suffix(f'.{os.getpid()}.tmp')
        with temporary.open('wb') as handle:
            pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, cache_path)
        self._metadata_history_cache[max_seq_len] = result
        return result

    def _new_history_model(
        self, model_name: str, field_dims: list[int], metadata_dims: dict[str, int],
        config: dict[str, Any], max_seq_len: int,
    ) -> torch.nn.Module:
        hp = config['hyperparameters']
        common = dict(
            field_dims=field_dims, metadata_dims=metadata_dims,
            embedding_dim=int(hp['embedding_dim']),
            hidden_dim=min(128, max(32, int(hp['embedding_dim']) * 4)),
            dropout=float(hp['dropout']),
        )
        if model_name == 'din':
            return CandidateAwareDIN(**common)
        if model_name == 'sasrec_meta':
            return MetadataSASRec(
                **common, max_seq_len=max_seq_len, num_heads=2,
                num_layers=2 if max_seq_len >= 50 else 1,
            )
        raise ValueError(f'unsupported history model: {model_name}')

    @staticmethod
    def _history_forward(
        model: torch.nn.Module, X: np.ndarray, sequence: dict[str, np.ndarray],
        history_indices: np.ndarray, candidate_indices: np.ndarray, *, device: str,
    ) -> torch.Tensor:
        history = {
            name: torch.as_tensor(sequence[name][history_indices], dtype=torch.long, device=device)
            for name in ('item', 'author', 'tag', 'duration')
        }
        metadata = {
            name: torch.as_tensor(sequence[f'candidate_{name}'][candidate_indices], dtype=torch.long, device=device)
            for name in ('item', 'author', 'tag', 'duration')
        }
        mask = torch.as_tensor(sequence['mask'][history_indices], dtype=torch.bool, device=device)
        candidate_x = torch.as_tensor(X[candidate_indices], dtype=torch.long, device=device)
        return model(candidate_x, history, mask, metadata)

    def _predict_history_model(
        self, model: torch.nn.Module, X: np.ndarray, sequence: dict[str, np.ndarray],
        *, batch_size: int, device: str,
    ) -> np.ndarray:
        model.eval()
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            all_indices = np.arange(len(X), dtype=np.int64)
            for start in range(0, len(X), batch_size):
                index = all_indices[start:start + batch_size]
                outputs.append(self._history_forward(
                    model, X, sequence, index, index, device=device
                ).cpu().numpy())
        scores = np.concatenate(outputs)
        if not np.all(np.isfinite(scores)):
            raise RuntimeError('history model produced non-finite predictions')
        return scores

    def _mine_history_negatives(
        self, model: torch.nn.Module, X: np.ndarray, sequence: dict[str, np.ndarray],
        positive: np.ndarray, pool: np.ndarray, keep: int, *, batch_size: int,
        device: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Select the model's highest-scoring candidates from sampled user pools."""
        selected = np.empty((len(positive), keep), dtype=np.int64)
        was_training = model.training
        model.eval()
        group_batch = max(1, batch_size // pool.shape[1])
        with torch.no_grad():
            for start in range(0, len(positive), group_batch):
                p = positive[start:start + group_batch]
                candidates = pool[start:start + group_batch]
                history_indices = np.repeat(p, candidates.shape[1])
                candidate_indices = candidates.reshape(-1)
                scores = self._history_forward(
                    model, X, sequence, history_indices, candidate_indices, device=device
                ).reshape(len(p), candidates.shape[1])
                top = torch.topk(scores, k=keep, dim=1).indices.cpu().numpy()
                selected[start:start + len(p)] = np.take_along_axis(candidates, top, axis=1)
        model.train(was_training)
        return np.repeat(positive, keep), selected.reshape(-1)

    def _run_history_model(
        self, config: dict[str, Any], checkpoint: Path, model_name: str,
    ) -> dict[str, Any]:
        hp = config['hyperparameters']
        max_seq_len = int(hp['sequence_length'])
        sequence, metadata_dims = self._metadata_history_data(max_seq_len)
        enc, field_dims = self._neural_data(config)
        Xtr, ytr, users = enc['train']
        Xva, yva, valid_users = enc['valid']
        original_Xtr = self._encoded_for(config)[0]['train'][0]
        seed = int(hp['seed'])
        batch_size = min(int(hp['batch_size']), 2048 if model_name == 'din' else 512)
        device = self._device()
        self._set_torch_seed(seed)
        model = self._new_history_model(
            model_name, field_dims, metadata_dims, config, max_seq_len
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(hp['learning_rate']), weight_decay=float(hp['l2'])
        )
        rng = np.random.default_rng(seed)
        best_score, best_state, best_epoch, bad = (-1.0, None, 0, 0)
        started = time.monotonic()
        effective_epochs = min(
            int(hp['epochs']), 10 if config['training_objective'] == 'bce' else 6
        )
        for epoch in range(1, effective_epochs + 1):
            model.train()
            losses: list[float] = []
            objective = config['training_objective']
            if objective in {'bpr', 'group_softmax'}:
                strategy = hp['negative_sampling_strategy']
                match = {'random': None, 'same_tab': original_Xtr[:, 3],
                         'same_author': original_Xtr[:, 2]}[strategy]
                keep = int(hp['negatives_per_positive'])
                hard_pool = int(hp['hard_negative_pool_size'])
                if hard_pool > 0:
                    positive, sampled = build_group_softmax_indices(
                        users, ytr, rng, max(keep, hard_pool), match
                    )
                    if len(positive) > 100_000:
                        positive, sampled = positive[:100_000], sampled[:100_000]
                    positive, negative_flat = self._mine_history_negatives(
                        model, Xtr, sequence['train'], positive, sampled, keep,
                        batch_size=batch_size, device=device,
                    )
                    if objective == 'group_softmax':
                        positive = positive.reshape(-1, keep)[:, 0]
                        negative = negative_flat.reshape(-1, keep)
                    else:
                        negative = negative_flat
                elif objective == 'group_softmax':
                    positive, negative = build_group_softmax_indices(
                        users, ytr, rng, keep, match
                    )
                else:
                    positive, negative = self._bpr_pairs(config, original_Xtr, users, ytr, rng)
                for start in range(0, len(positive), batch_size):
                    p = positive[start:start + batch_size]
                    n = negative[start:start + batch_size]
                    optimizer.zero_grad(set_to_none=True)
                    pos_score = self._history_forward(
                        model, Xtr, sequence['train'], p, p, device=device
                    )
                    if objective == 'group_softmax':
                        flat = n.reshape(-1)
                        repeated = np.repeat(p, n.shape[1])
                        neg_score = self._history_forward(
                            model, Xtr, sequence['train'], repeated, flat, device=device
                        ).reshape(len(p), -1)
                        logits = torch.cat((pos_score[:, None], neg_score), dim=1)
                        loss = torch.nn.functional.cross_entropy(
                            logits, torch.zeros(len(p), dtype=torch.long, device=device)
                        )
                    else:
                        neg_score = self._history_forward(
                            model, Xtr, sequence['train'], p, n, device=device
                        )
                        loss = -torch.nn.functional.logsigmoid(pos_score - neg_score).mean()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                    losses.append(float(loss.detach().cpu()))
            elif objective == 'bce':
                indices = rng.permutation(len(ytr))
                for start in range(0, len(indices), batch_size):
                    batch = indices[start:start + batch_size]
                    optimizer.zero_grad(set_to_none=True)
                    logits = self._history_forward(
                        model, Xtr, sequence['train'], batch, batch, device=device
                    )
                    target = torch.as_tensor(ytr[batch], dtype=torch.float32, device=device)
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                    losses.append(float(loss.detach().cpu()))
            else:
                raise ValueError(f'{model_name} does not support objective={objective!r}')
            scores = self._predict_history_model(
                model, Xva, sequence['valid'], batch_size=batch_size, device=device
            )
            metrics = self.evaluate_mod.evaluate(valid_users, yva, scores)
            current = self._validation_score(metrics, config)
            print(f"    {model_name} epoch {epoch:02d} | loss={np.mean(losses):.6f} | selection={config['hyperparameters'].get('validation_metric', 'primary')}:{current:.6f} | primary={float(metrics['primary']):.6f} | nDCG@5={float(metrics['nDCG@5']):.6f} | best={max(best_score, current):.6f}", flush=True)
            if current > best_score + 1e-5:
                best_score, best_epoch, bad = current, epoch, 0
                best_state = self._cpu_state(model)
            else:
                bad += 1
                if bad >= int(hp['patience']):
                    break
        if best_state is None:
            raise RuntimeError(f'{model_name} training produced no checkpoint')
        model.load_state_dict(best_state)
        scores = self._predict_history_model(
            model, Xva, sequence['valid'], batch_size=batch_size, device=device
        )
        valid = self.evaluate_mod.evaluate(valid_users, yva, scores)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        self._write_validation_slices(checkpoint, scores)
        torch.save({
            'model_name': model_name, 'field_dims': field_dims,
            'metadata_dims': metadata_dims, 'max_seq_len': max_seq_len,
            'state_dict': best_state, 'best_epoch': best_epoch,
        }, checkpoint)
        return {**self._metrics(valid), 'best_epoch': best_epoch,
                'runtime_seconds': float(time.monotonic() - started)}

    def _finalize_history_model(
        self, config: dict[str, Any], checkpoint: Path, output: Path, model_name: str,
    ) -> dict[str, Any]:
        saved = self._torch_load(checkpoint)
        max_seq_len = int(saved['max_seq_len'])
        sequence, metadata_dims = self._metadata_history_data(max_seq_len)
        enc, field_dims = self._neural_data(config)
        if field_dims != [int(value) for value in saved['field_dims']]:
            raise RuntimeError(f'{model_name} categorical vocabulary mismatch')
        if metadata_dims != {key: int(value) for key, value in saved['metadata_dims'].items()}:
            raise RuntimeError(f'{model_name} metadata vocabulary mismatch')
        device = self._device()
        model = self._new_history_model(
            model_name, field_dims, metadata_dims, config, max_seq_len
        ).to(device)
        model.load_state_dict(saved['state_dict'])
        scores = self._predict_history_model(
            model, enc['test'][0], sequence['test'],
            batch_size=min(int(config['hyperparameters']['batch_size']),
                           2048 if model_name == 'din' else 512), device=device,
        )
        return self._write_final(scores, output)

    def _run_din(self, config: dict[str, Any], checkpoint: Path) -> dict[str, Any]:
        return self._run_history_model(config, checkpoint, 'din')

    def _finalize_din(self, config: dict[str, Any], checkpoint: Path, output: Path) -> dict[str, Any]:
        return self._finalize_history_model(config, checkpoint, output, 'din')

    def _run_sasrec_meta(self, config: dict[str, Any], checkpoint: Path) -> dict[str, Any]:
        return self._run_history_model(config, checkpoint, 'sasrec_meta')

    def _finalize_sasrec_meta(self, config: dict[str, Any], checkpoint: Path, output: Path) -> dict[str, Any]:
        return self._finalize_history_model(config, checkpoint, output, 'sasrec_meta')

    def _new_sasrec(self, num_videos: int, config: dict[str, Any], max_seq_len: int) -> SASRecCandidateScorer:
        hp = config['hyperparameters']
        hidden_dim = min(128, max(32, int(hp['embedding_dim']) * 4))
        return SASRecCandidateScorer(num_videos=num_videos, hidden_dim=hidden_dim, max_seq_len=max_seq_len, num_heads=2, num_layers=2, dropout=float(hp['dropout']))

    @staticmethod
    def _predict_sasrec(model: SASRecCandidateScorer, data: dict[str, np.ndarray], *, batch_size: int, device: str) -> np.ndarray:
        model.eval()
        outputs = []
        with torch.no_grad():
            for start in range(0, len(data['candidate']), batch_size):
                sl = slice(start, start + batch_size)
                history = torch.as_tensor(data['history'][sl], dtype=torch.long, device=device)
                mask = torch.as_tensor(data['mask'][sl], dtype=torch.bool, device=device)
                candidate = torch.as_tensor(data['candidate'][sl], dtype=torch.long, device=device)
                outputs.append(model(history, mask, candidate).detach().cpu().numpy())
        scores = np.concatenate(outputs)
        if not np.all(np.isfinite(scores)):
            raise RuntimeError('sasrec produced non-finite predictions')
        return scores

    def _run_sasrec(self, config: dict[str, Any], checkpoint: Path) -> dict[str, Any]:
        max_seq_len = int(config['hyperparameters']['sequence_length'])
        seq, num_videos = self._sasrec_data(max_seq_len)
        enc, _ = self._encoded
        Xtr, ytr, users = enc['train']
        _, yva, valid_users = enc['valid']
        hp = config['hyperparameters']
        seed = int(hp['seed'])
        batch_size = min(int(hp['batch_size']), 4096)
        device = self._device()
        self._set_torch_seed(seed)
        model = self._new_sasrec(num_videos, config, max_seq_len).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(hp['learning_rate']), weight_decay=float(hp['l2']))
        rng = np.random.default_rng(seed)
        best_score, best_state, best_epoch, bad = (-1.0, None, 0, 0)
        started = time.monotonic()
        effective_epochs = min(
            int(hp['epochs']), 10 if config['training_objective'] == 'bce' else 6
        )
        for epoch in range(1, effective_epochs + 1):
            model.train()
            losses = []
            if config['training_objective'] == 'bpr':
                positive, negative = self._bpr_pairs(config, Xtr, users, ytr, rng)
                for start in range(0, len(positive), batch_size):
                    sl = slice(start, start + batch_size)
                    p, n = (positive[sl], negative[sl])
                    history = torch.as_tensor(seq['train']['history'][p], dtype=torch.long, device=device)
                    mask = torch.as_tensor(seq['train']['mask'][p], dtype=torch.bool, device=device)
                    pos_candidate = torch.as_tensor(seq['train']['candidate'][p], dtype=torch.long, device=device)
                    neg_candidate = torch.as_tensor(seq['train']['candidate'][n], dtype=torch.long, device=device)
                    optimizer.zero_grad(set_to_none=True)
                    pos_score = model(history, mask, pos_candidate)
                    neg_score = model(history, mask, neg_candidate)
                    loss = -torch.nn.functional.logsigmoid(pos_score - neg_score).mean()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                    losses.append(float(loss.detach().cpu()))
            elif config['training_objective'] == 'bce':
                indices = rng.permutation(len(ytr))
                for start in range(0, len(indices), batch_size):
                    batch = indices[start:start + batch_size]
                    history = torch.as_tensor(seq['train']['history'][batch], dtype=torch.long, device=device)
                    mask = torch.as_tensor(seq['train']['mask'][batch], dtype=torch.bool, device=device)
                    candidate = torch.as_tensor(seq['train']['candidate'][batch], dtype=torch.long, device=device)
                    target = torch.as_tensor(ytr[batch], dtype=torch.float32, device=device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(history, mask, candidate), target)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                    losses.append(float(loss.detach().cpu()))
            else:
                raise ValueError(f"sasrec supports only bce/bpr, got {config['training_objective']!r}")
            scores = self._predict_sasrec(model, seq['valid'], batch_size=batch_size, device=device)
            metrics = self.evaluate_mod.evaluate(valid_users, yva, scores)
            current = self._validation_score(metrics, config)
            print(f"    sasrec epoch {epoch:02d} | loss={np.mean(losses):.6f} | selection={config['hyperparameters'].get('validation_metric', 'primary')}:{current:.6f} | primary={float(metrics['primary']):.6f} | nDCG@5={float(metrics['nDCG@5']):.6f} | best={max(best_score, current):.6f}", flush=True)
            if current > best_score + 1e-05:
                best_score, best_epoch, bad = (current, epoch, 0)
                best_state = self._cpu_state(model)
            else:
                bad += 1
                if bad >= int(hp['patience']):
                    break
        if best_state is None:
            raise RuntimeError('sasrec training produced no checkpoint')
        model.load_state_dict(best_state)
        scores = self._predict_sasrec(model, seq['valid'], batch_size=batch_size, device=device)
        valid = self.evaluate_mod.evaluate(valid_users, yva, scores)
        self._write_validation_slices(checkpoint, scores)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'num_videos': num_videos, 'max_seq_len': max_seq_len, 'state_dict': best_state, 'best_epoch': best_epoch}, checkpoint)
        return {'GAUC': float(valid['GAUC']), 'nDCG@5': float(valid['nDCG@5']), 'primary': float(valid['primary']), 'best_epoch': int(best_epoch), 'ensemble_size': 1, 'runtime_seconds': float(time.monotonic() - started)}

    def _finalize_sasrec(self, config: dict[str, Any], checkpoint: Path, output: Path) -> dict[str, Any]:
        saved = self._torch_load(checkpoint)
        max_seq_len = int(saved['max_seq_len'])
        seq, num_videos = self._sasrec_data(max_seq_len)
        if int(saved['num_videos']) != num_videos:
            raise RuntimeError('sasrec video vocabulary mismatch')
        device = self._device()
        model = self._new_sasrec(num_videos, config, max_seq_len).to(device)
        model.load_state_dict(saved['state_dict'])
        scores = self._predict_sasrec(model, seq['test'], batch_size=min(int(config['hyperparameters']['batch_size']), 4096), device=device)
        return self._write_final(scores, output)

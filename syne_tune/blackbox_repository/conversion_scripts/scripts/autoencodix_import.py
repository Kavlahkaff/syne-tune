import json
from pathlib import Path

import numpy as np
import pandas as pd

from syne_tune.blackbox_repository.blackbox_tabular import BlackboxTabular, serialize
from syne_tune.blackbox_repository.conversion_scripts.blackbox_recipe import (
    BlackboxRecipe,
)
from syne_tune.blackbox_repository.conversion_scripts.scripts import (
    default_metric,
    metric_elapsed_time,
    time_attr,
)
from syne_tune.blackbox_repository.conversion_scripts.utils import repository_path
from syne_tune.config_space import choice, loguniform, randint, uniform
from syne_tune.util import catchtime

RESULTS_ROOT = Path("/Users/lucathale-bombien/autoencodix_results")

METRIC_ELAPSED_TIME = "metric_elapsed_time"
METRIC_RECON_LOSS = "metric_valid_recon_loss"
METRIC_DOWNSTREAM = "metric_avg_ml_task_performance"
TIME_ATTR = "epoch"

OBJECTIVES = [METRIC_ELAPSED_TIME, METRIC_RECON_LOSS, METRIC_DOWNSTREAM]

ONTOLOGY_ARCHITECTURES = {"ontix"}

_SHARED_HPS = {
    "k_filter": choice([128, 256, 512, 1024, 2048, 4096]),
    "n_layers": choice([2, 3, 4]),
    "enc_factor": uniform(1, 4),
    "batch_size": choice([32, 64, 128, 256]),
    "learning_rate": loguniform(1e-5, 1e-1),
    "drop_p": uniform(0, 0.9),
    "weight_decay": loguniform(1e-5, 1e-1),
    "latent_dim": choice([2, 4, 8, 16, 32, 64]),
}

ARCHITECTURE_CONFIG_SPACES = {
    "vanillix": {**_SHARED_HPS},
    "varix": {**_SHARED_HPS, "beta": loguniform(0.001, 10)},
    "ontix": {**_SHARED_HPS},
    "disentanglix": {
        **_SHARED_HPS,
        "beta_mi": loguniform(0.001, 10.0),
        "beta_tc": loguniform(0.1, 10000),
        "beta_dimKL": loguniform(0.001, 10.0),
    },
}


def _seed_parent_map(modalities_dir: Path, architecture: str) -> dict[str, Path]:
    """
    Return {ontology_suffix: seed_parent_dir} for one modalities directory.

    For ontology architectures, subdirs are either seed dirs (no ontology level)
    or ontology dirs (one level deeper). For all others, maps "" to modalities_dir.
    """
    if architecture not in ONTOLOGY_ARCHITECTURES:
        return {"": modalities_dir}

    subdirs = [d for d in sorted(modalities_dir.iterdir()) if d.is_dir()]
    if not subdirs or any(d.name.startswith("seed_") for d in subdirs):
        return {"": modalities_dir}

    return {d.name: d for d in subdirs}


def load_json_results(results_root: Path) -> dict[tuple, list[dict]]:
    """
    Walk results_root/<arch>/<dataset>/<modalities>/[<ontology>/]seed_<n>/<run>.json
    and return a mapping (architecture, dataset, modalities) → list[record].
    """
    records: dict[tuple, list[dict]] = {}
    n_files = 0

    for arch_dir in sorted(results_root.iterdir()):
        if not arch_dir.is_dir():
            continue
        architecture = arch_dir.name.lower()

        for dataset_dir in sorted(arch_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue

            for modalities_dir in sorted(dataset_dir.iterdir()):
                if not modalities_dir.is_dir():
                    continue

                for ontology_suffix, seed_parent in _seed_parent_map(
                    modalities_dir, architecture
                ).items():
                    task_modalities = (
                        f"{modalities_dir.name}_{ontology_suffix}"
                        if ontology_suffix
                        else modalities_dir.name
                    )
                    key = (architecture, dataset_dir.name, task_modalities)
                    records.setdefault(key, [])

                    for seed_dir in sorted(seed_parent.iterdir()):
                        if not seed_dir.is_dir() or not seed_dir.name.startswith(
                            "seed_"
                        ):
                            print(f"  [SKIP] unexpected directory: {seed_dir}")
                            continue
                        if not seed_dir.name[len("seed_") :].isdigit():
                            print(f"  [SKIP] non-integer seed directory: {seed_dir}")
                            continue

                        for json_file in sorted(seed_dir.glob("*.json")):
                            with open(json_file) as fh:
                                records[key].append(json.load(fh))
                            n_files += 1

    print(
        f"  Found {n_files} JSON files across "
        f"{len(records)} (architecture, dataset, modalities) combos"
    )
    if empty := [k for k, v in records.items() if not v]:
        print(f"  WARNING: 0 files found for {len(empty)} combo(s):")
        for k in empty:
            print(f"    {k}")

    return records


def _extract_hps(record: dict, hp_keys: list[str]) -> dict:
    """Extract hyperparameter values from a record's HYPERPARAMETERS block."""
    return {k: record["HYPERPARAMETERS"][k] for k in hp_keys}


def _hp_key(hp_row: dict) -> tuple:
    return tuple(sorted(hp_row.items()))


def _build_arch_index(
    all_runs: list[dict], hp_keys: list[str]
) -> tuple[pd.DataFrame, dict[tuple, int], list[int], int]:
    """
    Single pass over all runs for one architecture to build:
      - union_hp_df : deduplicated HP DataFrame
      - hp_index    : hp_key → row index
      - seed_list   : sorted unique seeds
    """
    seen: dict[tuple, dict] = {}
    hp_order: list[tuple] = []
    seeds: set[int] = set()
    max_epochs = 300

    for record in all_runs:
        hp_row = _extract_hps(record, hp_keys)
        key = _hp_key(hp_row)
        if key not in seen:
            seen[key] = hp_row
            hp_order.append(key)
        seeds.add(record["SEED"])

    union_hp_df = pd.DataFrame([seen[k] for k in hp_order]).reset_index(drop=True)
    return (
        union_hp_df,
        {k: i for i, k in enumerate(hp_order)},
        sorted(seeds),
        max_epochs,
    )


def _fill_objectives(
    records: list[dict],
    hp_keys: list[str],
    hp_index: dict[tuple, int],
    seed_to_idx: dict[int, int],
    shape: tuple[int, int, int],
) -> np.ndarray:
    """
    Build objectives_evaluations array of shape (num_hps, num_seeds, max_epochs, 3).

    - valid_recon_loss         : filled at every epoch
    - runtime_seconds          : filled only at the final epoch
    - avg_ml_task_performance  : filled only at the final epoch
    """
    obj_array = np.full((*shape, len(OBJECTIVES)), np.nan, dtype=np.float64)
    obj_idx = {name: i for i, name in enumerate(OBJECTIVES)}

    for record in records:
        hp_idx = hp_index[_hp_key(_extract_hps(record, hp_keys))]
        s_idx = seed_to_idx[record["SEED"]]
        loss_dict: dict[str, float] = record["loss_per_epoch"]
        final = len(loss_dict) - 1

        for epoch_str, recon_loss in sorted(
            loss_dict.items(), key=lambda kv: int(kv[0])
        ):
            e = int(epoch_str)
            obj_array[hp_idx, s_idx, e, obj_idx[METRIC_RECON_LOSS]] = recon_loss
            if e == final:
                obj_array[hp_idx, s_idx, e, obj_idx[METRIC_ELAPSED_TIME]] = record[
                    "RUNTIME_SECONDS"
                ]
                obj_array[hp_idx, s_idx, e, obj_idx[METRIC_DOWNSTREAM]] = record[
                    "AVG_ML_TASK_PERFORMANCE"
                ]

    return obj_array


def generate_autoencodix_from_json(results_root: Path = RESULTS_ROOT) -> None:
    with catchtime("loading JSON results"):
        all_records = load_json_results(results_root)

    if not all_records:
        raise FileNotFoundError(
            f"No JSON run files found under '{results_root}'. "
            "Expected layout: <arch>/<dataset>/<modalities>/[<ontology>/]seed_<n>/<run>.json"
        )

    # Group by architecture
    arch_tasks: dict[str, dict[str, list[dict]]] = {}
    for (arch, dataset, modalities), records in all_records.items():
        arch_tasks.setdefault(arch, {})[f"{dataset}_{modalities}"] = records

    for architecture, task_records in arch_tasks.items():
        config_space = ARCHITECTURE_CONFIG_SPACES.get(architecture)
        if config_space is None:
            print(f"[SKIP] No config space for '{architecture}'.")
            continue

        hp_keys = list(config_space.keys())
        all_runs = [r for recs in task_records.values() for r in recs]
        print(
            f"\nArchitecture: {architecture}  ({len(all_runs)} runs, {len(task_records)} tasks)"
        )

        with catchtime("  building union HP index"):
            union_hp_df, hp_index, seed_list, max_epochs = _build_arch_index(
                all_runs, hp_keys
            )

        seed_to_idx = {s: i for i, s in enumerate(seed_list)}
        num_hps, num_seeds = len(union_hp_df), len(seed_list)
        fidelity_values = np.arange(1, max_epochs + 1)
        fidelity_space = {TIME_ATTR: randint(lower=300, upper=300)}

        print(f"Union HP configs: {num_hps}  |  Seeds: {seed_list} ")

        bb_dict: dict[str, BlackboxTabular] = {}
        for task_name, records in task_records.items():
            print(f"  Converting {len(records):5d} runs  →  task={task_name}")
            with catchtime(f"    filling {task_name}"):
                obj_array = _fill_objectives(
                    records,
                    hp_keys,
                    hp_index,
                    seed_to_idx,
                    shape=(num_hps, num_seeds, max_epochs),
                )
            bb_dict[task_name] = BlackboxTabular(
                hyperparameters=union_hp_df.copy(),
                configuration_space=config_space,
                fidelity_space=fidelity_space,
                objectives_evaluations=obj_array,
                fidelity_values=fidelity_values,
                objectives_names=OBJECTIVES,
            )

        blackbox_name = f"autoencodix_{architecture}"
        with catchtime(f"  serializing {blackbox_name}"):
            serialize(
                bb_dict=bb_dict,
                path=repository_path / blackbox_name,
                metadata={
                    metric_elapsed_time: METRIC_ELAPSED_TIME,
                    default_metric: METRIC_DOWNSTREAM,
                    time_attr: TIME_ATTR,
                },
            )
        print(f"  Saved {blackbox_name}  tasks: {sorted(bb_dict)}")


class AutoencodixJsonRecipe(BlackboxRecipe):
    def __init__(self, architecture: str):
        super().__init__(
            name=f"autoencodix_{architecture}",
            cite_reference=(
                "AUTOENCODIX: A generalized and versatile framework to train and "
                "evaluate autoencoders for biological representation learning and "
                "beyond. Maximilian Joas, Neringa Jurenaite, Dusan Prascevic, "
                "Nico Scherf, Jan Ewald. BioRxiv, 2024."
            ),
        )
        self.architecture = architecture

    def _generate_on_disk(self) -> None:
        generate_autoencodix_from_json()


AutoEncodixVanillixJsonRecipe = type(
    "AutoEncodixVanillixJsonRecipe",
    (AutoencodixJsonRecipe,),
    {"__init__": lambda self: AutoencodixJsonRecipe.__init__(self, "vanillix")},
)
AutoEncodixVarixJsonRecipe = type(
    "AutoEncodixVarixJsonRecipe",
    (AutoencodixJsonRecipe,),
    {"__init__": lambda self: AutoencodixJsonRecipe.__init__(self, "varix")},
)
AutoEncodixOntixJsonRecipe = type(
    "AutoEncodixOntixJsonRecipe",
    (AutoencodixJsonRecipe,),
    {"__init__": lambda self: AutoencodixJsonRecipe.__init__(self, "ontix")},
)
AutoEncodixDisentanglixJsonRecipe = type(
    "AutoEncodixDisentanglixJsonRecipe",
    (AutoencodixJsonRecipe,),
    {"__init__": lambda self: AutoencodixJsonRecipe.__init__(self, "disentanglix")},
)


if __name__ == "__main__":
    generate_autoencodix_from_json()

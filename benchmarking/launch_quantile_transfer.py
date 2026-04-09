from __future__ import annotations

import argparse
import logging

from syne_tune import StoppingCriterion, Tuner
from syne_tune.backend.simulator_backend.simulator_callback import SimulatorCallback
from syne_tune.blackbox_repository import BlackboxRepositoryBackend, load_blackbox
from syne_tune.blackbox_repository.blackbox_tabular import BlackboxTabular
from syne_tune.experiments import load_experiment
from syne_tune.optimizer.schedulers.single_objective_scheduler import (
    SingleObjectiveScheduler,
)
from syne_tune.optimizer.schedulers.transfer_learning.quantile_based.quantile_based_searcher import (
    QuantileBasedSurrogateSearcher,
)
from utils import load_transfer_learning_evaluations, METRIC_MODE, METRIC_DOWNSTREAM, METRIC_RECON_LOSS, METRIC_ELAPSED_TIME
import numpy as np
logger = logging.getLogger(__name__)

def make_tuner_name(args) -> str:
    # --- metric abbreviation ---
    metric_short = {
        METRIC_DOWNSTREAM:   "ds",
        METRIC_RECON_LOSS:   "rl",
        METRIC_ELAPSED_TIME: "et",
    }.get(args.metric, args.metric[:4])

    # --- extra-architectures suffix (sorted for stability) ---
    extra = "+".join(sorted(args.extra_architectures)) if args.extra_architectures else "none"

    # --- boolean flags as compact bits ---
    flags = "".join([
        "A" if args.all_datasets    else "a",   # All-datasets on/off
        "X" if args.cross_arch_only else "x",   # Cross-arch-only on/off
    ])

    tuner_name = (
        f"ZeroShot"
        f"-{args.architecture}"          # target architecture
        f"-{args.test_task}"             # held-out task
        f"-{metric_short}"               # optimisation metric
        f"-{flags}"                      # three boolean switches
        f"-x{extra}"                     # extra source architectures
        f"-r{args.seed}"                 # random seed
    )
    return tuner_name


def run(
    architecture: str,
    test_task: str,
    metric: str,
    do_minimize: bool,
    normalization: str,
    max_fit_samples: int,
    max_wallclock_time: float,
    n_workers: int,
    random_seed: int,
    same_dataset_only: bool = True,
    extra_architectures: list[str] | None = None,
    cross_arch_only: bool = False,
) -> None:
    np.random.seed(args.seed)
    blackbox_name = f"autoencodix_{architecture}"
    bb_dict = load_blackbox(blackbox_name, local_files_only=True)
    available_tasks = sorted(bb_dict)
    logger.info("Blackbox '%s' — tasks: %s", blackbox_name, available_tasks)

    if test_task not in available_tasks:
        raise ValueError(
            f"test_task='{test_task}' not found. Available: {available_tasks}"
        )

    # Load extra-architecture blackboxes once before the cycle loop
    extra_bb_dicts: dict[str, dict[str, BlackboxTabular]] = {}
    for arch in (extra_architectures or []):
        if arch == architecture:
            logger.warning(
                "--extra-architecture '%s' is the same as the target; skipping.", arch
            )
            continue
        arch_bb_name = f"autoencodix_{arch}"
        logger.info("Loading extra-architecture blackbox '%s' …", arch_bb_name)
        extra_bb_dicts[arch] = load_blackbox(arch_bb_name, local_files_only=True)


    transfer_learning_evaluations = load_transfer_learning_evaluations(
        blackbox_name=blackbox_name,
        test_task=test_task,
        metric=metric,
        bb_dict=bb_dict,
        same_dataset_only=same_dataset_only,
        extra_bb_dicts=extra_bb_dicts if extra_bb_dicts else None,
        cross_arch_only=cross_arch_only,
    )

    searcher = QuantileBasedSurrogateSearcher(
        config_space=bb_dict[test_task].configuration_space,
        transfer_learning_evaluations=transfer_learning_evaluations,
        normalization=normalization,
        max_fit_samples=max_fit_samples,
        random_seed=random_seed,
    )

    scheduler = SingleObjectiveScheduler(
        config_space=bb_dict[test_task].configuration_space,
        searcher=searcher,
        metric=metric,
        do_minimize=do_minimize,
        random_seed=random_seed,
    )

    stop_criterion = StoppingCriterion(max_num_trials_finished=100)

    trial_backend = BlackboxRepositoryBackend(
        blackbox_name=blackbox_name,
        elapsed_time_attr=METRIC_ELAPSED_TIME,
        dataset=test_task,
        surrogate="KNeighborsRegressor",
        surrogate_kwargs={"n_neighbors": 1},
    )
    name = make_tuner_name(args)
    tuner = Tuner(
        tuner_name=name,
        suffix_tuner_name=False,
        trial_backend=trial_backend,
        scheduler=scheduler,
        stop_criterion=stop_criterion,
        n_workers=n_workers,
        sleep_time=0,
        callbacks=[SimulatorCallback()],
        metadata={
            "seed": random_seed,
            "algorithm": args.searcher,
            "benchmark": architecture + "-" + test_task,
        },
    )

    tuner.run()

    # ── Final report ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Overall best result (quantile-based surrogate transfer)")
    print(f"Architecture  : {architecture}")
    print(f"Test task     : {test_task}")
    print(f"Metric        : {metric}  ({'min' if do_minimize else 'max'})")
    print(f"Normalization : {normalization}")
    if extra_architectures:
        print(f"Extra sources : {extra_architectures}")
    print(f"{'='*60}")

    best_experiment = load_experiment(tuner.name)
    best_config     = best_experiment.best_config()
    target_cs       = bb_dict[test_task].configuration_space
    print("\n".join(
        f"  {k}: {v}" for k, v in best_config.items()
        if k in target_cs
    ))

    best_experiment.plot()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Quantile-based surrogate transfer-learning tuner for autoencodix."
    )
    p.add_argument(
        "--architecture",
        default="ontix",
        choices=["vanillix", "varix", "ontix", "disentanglix"],
    )
    p.add_argument(
        "--test-task",
        default=None,
        help="Task held out as the optimisation target (default: first task).",
    )
    p.add_argument(
        "--metric",
        default=METRIC_DOWNSTREAM,
        choices=[METRIC_DOWNSTREAM, METRIC_RECON_LOSS, METRIC_ELAPSED_TIME],
    )
    p.add_argument(
        "--normalization",
        default="gaussian",
        choices=["gaussian", "standard"],
        help=(
            "'gaussian' applies rank transform then Gaussian inverse CDF (default). "
            "'standard' subtracts mean and divides by std — faster but weaker."
        ),
    )
    p.add_argument(
        "--max-fit-samples",
        type=int,
        default=100000,
        help="Maximum number of samples used to fit the surrogate (default: 100k).",
    )
    p.add_argument(
        "--max-wallclock-time",
        type=float,
        default=3600 * 4,
        help="Simulated wall-clock budget in seconds (default: 4 h).",
    )
    p.add_argument("--n-workers",  type=int, default=1)
    p.add_argument("--seed",      type=int, default=0)
    p.add_argument("--log-level", default="INFO")
    p.add_argument(
        "--all-datasets",
        action="store_true",
        default=False,
        help=(
            "Use source tasks from ALL datasets, not just the same dataset as "
            "the test task. Default: same-dataset only."
        ),
    )
    p.add_argument(
        "--cross-arch-only",
        action="store_true",
        default=False,
        help=(
            "Use only cross-architecture source tasks; exclude all same-architecture "
            "tasks. Requires --extra-architectures to be set."
        ),
    )
    p.add_argument(
        "--extra-architectures",
        nargs="*",
        default=["vanillix"],
        choices=["vanillix", "varix", "ontix", "disentanglix"],
        help=(
            "Additional architectures to use as transfer-learning sources. "
            "Hyperparameters not shared with the target architecture are left "
            "unconstrained (sampled freely). "
            "Example: --extra-architectures varix vanillix"
        ),
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s – %(message)s",
    )

    if args.test_task is None:
        _bb = load_blackbox(f"autoencodix_{args.architecture}")
        args.test_task = sorted(_bb)[0]
        print(f"No --test-task given, defaulting to '{args.test_task}'")

    run(
        architecture=args.architecture,
        test_task=args.test_task,
        metric=args.metric,
        do_minimize=METRIC_MODE[args.metric],
        normalization=args.normalization,
        max_fit_samples=args.max_fit_samples,
        max_wallclock_time=args.max_wallclock_time,
        n_workers=args.n_workers,
        random_seed=args.seed,
        same_dataset_only=not args.all_datasets,
        extra_architectures=args.extra_architectures or None,
        cross_arch_only=args.cross_arch_only,
    )
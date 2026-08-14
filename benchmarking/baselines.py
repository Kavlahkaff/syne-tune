from dataclasses import dataclass
from pathlib import Path
from typing import Any

from syne_tune.blackbox_repository.simulated_tabular_backend import (
    BlackboxRepositoryBackend,
)
from syne_tune.optimizer.scheduler import TrialScheduler
from syne_tune.optimizer.schedulers.asha import AsynchronousSuccessiveHalving
from syne_tune.optimizer.schedulers.single_objective_scheduler import (
    SingleObjectiveScheduler,
)
from syne_tune.optimizer.schedulers.searchers.fmbo.fmbo_searcher import FMBOSearcher


@dataclass
class MethodArguments:
    config_space: dict
    metric: str
    mode: str
    random_seed: int
    time_attr: str
    points_to_evaluate: list[dict]
    max_t: int | None = None
    max_resource_attr: str | None = None
    use_surrogates: bool = False
    num_brackets: int | None = 1
    verbose: bool | None = False
    checkpoint_dir: str | None = None
    benchmark_name: str | None = None
    model: Any | None = None
    tokenizer: Any | None = None
    gpu_memory_utilization: float = 0.2


class Methods:
    # single fidelity
    BORE = "BORE"
    RS = "RS"
    TPE = "TPE"
    REA = "REA"
    BOTorch = "BOTorch"
    CQR = "CQR"
    BOHB = "BOHB"

    # optformer/fmbo
    OPT_RS = "OPT-RS"
    OPT_CQR = "OPT-CQR"
    OPT_REA = "OPT-REA"
    OPT_BORE = "OPT-BORE"
    OPT_TPE = "OPT-TPE"
    OPT_HEBO = "OPT-HEBO"
    OPT_CQR_TS = "OPT-CQR-TS"
    OPT_CQR_TS_5 = "OPT-CQR-TS-5"

    # multifidelity
    ASHA = "ASHA"
    ASHABORE = "ASHABORE"
    ASHACQR = "ASHACQR"


def _fmbo_scheduler(method_arguments, algorithm: str, n_sample_configurations: int = 1):
    return SingleObjectiveScheduler(
        config_space=method_arguments.config_space,
        metric=method_arguments.metric,
        do_minimize=method_arguments.mode == "min",
        random_seed=method_arguments.random_seed,
        searcher=FMBOSearcher(
            config_space=method_arguments.config_space,
            checkpoint_dir=Path(method_arguments.checkpoint_dir) if method_arguments.checkpoint_dir else "synetune/qwen3_80M_token_2B_lr_5e-3_bsz_16_seed_0",
            tokenizer_dir="synetune/bbo-pile-tokenizer",
            use_vllm=True,
            task_info={'name': method_arguments.benchmark_name,
                       'algorithm': algorithm,
                       'metric_names': "feval"},
            random_seed=method_arguments.random_seed,
            points_to_evaluate=method_arguments.points_to_evaluate,
            n_sample_configurations=n_sample_configurations,
            model=method_arguments.model,
            tokenizer=method_arguments.tokenizer,
            gpu_memory_utilization=method_arguments.gpu_memory_utilization,
        ),
    )


methods = {
    Methods.RS: lambda method_arguments: SingleObjectiveScheduler(
        config_space=method_arguments.config_space,
        searcher="random_search",
        metric=method_arguments.metric,
        do_minimize=method_arguments.mode == "min",
        random_seed=method_arguments.random_seed,
        searcher_kwargs={"points_to_evaluate": method_arguments.points_to_evaluate},
    ),
    Methods.BORE: lambda method_arguments: SingleObjectiveScheduler(
        config_space=method_arguments.config_space,
        searcher="bore",
        metric=method_arguments.metric,
        do_minimize=method_arguments.mode == "min",
        random_seed=method_arguments.random_seed,
        searcher_kwargs={"points_to_evaluate": method_arguments.points_to_evaluate},
    ),
    Methods.TPE: lambda method_arguments: SingleObjectiveScheduler(
        config_space=method_arguments.config_space,
        searcher="kde",
        metric=method_arguments.metric,
        do_minimize=method_arguments.mode == "min",
        random_seed=method_arguments.random_seed,
        searcher_kwargs={"points_to_evaluate": method_arguments.points_to_evaluate},
    ),
    Methods.CQR: lambda method_arguments: SingleObjectiveScheduler(
        config_space=method_arguments.config_space,
        searcher="cqr",
        metric=method_arguments.metric,
        do_minimize=method_arguments.mode == "min",
        random_seed=method_arguments.random_seed,
        searcher_kwargs={"points_to_evaluate": method_arguments.points_to_evaluate},
    ),
    Methods.BOTorch: lambda method_arguments: SingleObjectiveScheduler(
        config_space=method_arguments.config_space,
        searcher="botorch",
        metric=method_arguments.metric,
        do_minimize=method_arguments.mode == "min",
        random_seed=method_arguments.random_seed,
        searcher_kwargs={"points_to_evaluate": method_arguments.points_to_evaluate},
    ),
    Methods.REA: lambda method_arguments: SingleObjectiveScheduler(
        config_space=method_arguments.config_space,
        searcher="regularized_evolution",
        metric=method_arguments.metric,
        do_minimize=method_arguments.mode == "min",
        random_seed=method_arguments.random_seed,
        searcher_kwargs={"points_to_evaluate": method_arguments.points_to_evaluate},
    ),
    Methods.BOHB: lambda method_arguments: AsynchronousSuccessiveHalving(
        config_space=method_arguments.config_space,
        metric=method_arguments.metric,
        do_minimize=method_arguments.mode == "min",
        random_seed=method_arguments.random_seed,
        searcher="kde",
        time_attr=method_arguments.time_attr,
        searcher_kwargs={"points_to_evaluate": method_arguments.points_to_evaluate},
    ),
    Methods.ASHA: lambda method_arguments: AsynchronousSuccessiveHalving(
        config_space=method_arguments.config_space,
        metric=method_arguments.metric,
        do_minimize=method_arguments.mode == "min",
        random_seed=method_arguments.random_seed,
        searcher="random_search",
        time_attr=method_arguments.time_attr,
        searcher_kwargs={"points_to_evaluate": method_arguments.points_to_evaluate},
    ),
    Methods.ASHACQR: lambda method_arguments: AsynchronousSuccessiveHalving(
        config_space=method_arguments.config_space,
        metric=method_arguments.metric,
        do_minimize=method_arguments.mode == "min",
        random_seed=method_arguments.random_seed,
        searcher="cqr",
        time_attr=method_arguments.time_attr,
        searcher_kwargs={"points_to_evaluate": method_arguments.points_to_evaluate},
    ),
    Methods.ASHABORE: lambda method_arguments: AsynchronousSuccessiveHalving(
        config_space=method_arguments.config_space,
        metric=method_arguments.metric,
        do_minimize=method_arguments.mode == "min",
        random_seed=method_arguments.random_seed,
        searcher="bore",
        time_attr=method_arguments.time_attr,
        searcher_kwargs={"points_to_evaluate": method_arguments.points_to_evaluate},
    ),
    Methods.OPT_RS: lambda method_arguments: _fmbo_scheduler(method_arguments, "RS"),
    Methods.OPT_CQR: lambda method_arguments: _fmbo_scheduler(method_arguments, "CQR"),
    Methods.OPT_REA: lambda method_arguments: _fmbo_scheduler(method_arguments, "REA"),
    Methods.OPT_BORE: lambda method_arguments: _fmbo_scheduler(method_arguments, "BORE"),
    Methods.OPT_TPE: lambda method_arguments: _fmbo_scheduler(method_arguments, "TPE"),
    Methods.OPT_HEBO: lambda method_arguments: _fmbo_scheduler(method_arguments, "HEBO"),
    Methods.OPT_CQR_TS: lambda method_arguments: _fmbo_scheduler(method_arguments, "CQR", n_sample_configurations=50),
    Methods.OPT_CQR_TS_5: lambda method_arguments: _fmbo_scheduler(method_arguments, "CQR", n_sample_configurations=5),
}


if __name__ == "__main__":
    # Run a loop that initializes all schedulers on all benchmark to see if they all work
    from benchmarking.benchmarks import (
        benchmark_definitions,
    )

    print(f"Checking initialization of {list(methods.keys())[::-1]}")

    benchmarks = [
        "fcnet-protein",
        "nas201-cifar10",
        "lcbench-Fashion-MNIST",
        "tabrepo-RandomForest-2dplanes",
        "hpob_5636_3492",
    ]
    for benchmark_name in benchmarks:
        benchmark = benchmark_definitions[benchmark_name]
        backend = BlackboxRepositoryBackend(
            elapsed_time_attr=benchmark.elapsed_time_attr,
            blackbox_name=benchmark.blackbox_name,
            dataset=benchmark.dataset_name,
        )
        points_to_evaluate = [
            {
                k: v.sample() if hasattr(v, "sample") else v
                for k, v in backend.blackbox.configuration_space.items()
            }
            for _ in range(4)
        ]
        print(f"Checking initialization of {list(methods.keys())[::-1]}")
        for method_name, method_fun in list(methods.items())[::-1]:
            print(f"checking initialization of: {method_name}, {benchmark_name}")
            # if method_name != Methods.QHB_XGB:
            #     continue

            scheduler = method_fun(
                MethodArguments(
                    config_space=backend.blackbox.configuration_space,
                    metric=benchmark.metric,
                    mode=benchmark.mode,
                    random_seed=0,
                    max_t=max(backend.blackbox.fidelity_values),
                    time_attr=next(iter(backend.blackbox.fidelity_space.keys())),
                    use_surrogates=benchmark_name == "lcbench-Fashion-MNIST",
                    points_to_evaluate=points_to_evaluate,
                )
            )
            if isinstance(scheduler, TrialScheduler):
                print(scheduler.suggest())
                print(scheduler.suggest())
            else:
                print(scheduler.suggest(0))
                print(scheduler.suggest(1))

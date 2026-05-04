"""
This example shows how to run transfer learning methods on autoencodix blackboxes.
We demonstrate testing BoundingBox, QuantileTransfer, and ZeroShotTransfer methods.
"""
import logging
from syne_tune.blackbox_repository import (
    load_blackbox,
    BlackboxRepositoryBackend,
)

from syne_tune.backend.simulator_backend.simulator_callback import SimulatorCallback
from syne_tune import StoppingCriterion, Tuner
from benchmarking.utils import load_transfer_learning_evaluations, _sanitize_config_space

from syne_tune.optimizer.schedulers.single_objective_scheduler import SingleObjectiveScheduler
from syne_tune.optimizer.schedulers.transfer_learning.bounding_box import BoundingBox
from syne_tune.optimizer.schedulers.transfer_learning.quantile_based.quantile_based_searcher import QuantileBasedSurrogateSearcher
from syne_tune.optimizer.schedulers.transfer_learning.zero_shot import ZeroShotTransfer


def simulate_transfer_benchmark(blackbox_name, test_task, trial_backend, metric):
    bb_dict = load_blackbox(blackbox_name, local_files_only=True)
    config_space = bb_dict[test_task].configuration_space

    # Load transfer learning evaluations (we'll use the same dataset for transfer source)
    transfer_learning_evaluations = load_transfer_learning_evaluations(
        blackbox_name=blackbox_name,
        test_task=test_task,
        metric=metric,
        bb_dict=bb_dict,
        same_dataset_only=True,
    )

    # 1. Test BoundingBox
    def _make_inner_scheduler(new_config_space, metric, do_minimize, random_seed):
        clean_space = _sanitize_config_space(new_config_space)
        return SingleObjectiveScheduler(
            clean_space,
            do_minimize=do_minimize,
            searcher="random_search",
            metric=metric,
            random_seed=random_seed,
        )

    scheduler_bbox = BoundingBox(
        scheduler_fun=_make_inner_scheduler,
        config_space=config_space,
        metric=metric,
        do_minimize=True,
        num_hyperparameters_per_task=20,
        transfer_learning_evaluations=transfer_learning_evaluations,
        random_seed=42,
    )

    # 2. Test QuantileTransfer
    searcher_qt = QuantileBasedSurrogateSearcher(
        config_space=config_space,
        transfer_learning_evaluations=transfer_learning_evaluations,
        normalization="gaussian",
        max_fit_samples=100000,
        random_seed=42,
    )

    scheduler_qt = SingleObjectiveScheduler(
        config_space=config_space,
        searcher=searcher_qt,
        metric=metric,
        do_minimize=True,
        random_seed=42,
    )

    # 3. Test ZeroShotTransfer
    scheduler_zs = ZeroShotTransfer(
        config_space=config_space,
        metric=metric,
        do_minimize=True,
        transfer_learning_evaluations=transfer_learning_evaluations,
        use_surrogates=False,
        random_seed=42,
    )

    stop_criterion = StoppingCriterion(max_wallclock_time=20000)  # 1 simulated hour
    n_workers = 1

    # Run one of the methods to test (you can swap scheduler_bbox, scheduler_qt, scheduler_zs)
    print("Running ZeroShotTransfer...")
    tuner = Tuner(
        trial_backend=trial_backend,
        scheduler=scheduler_bbox, # Try `scheduler_bbox` or `scheduler_qt`  or `scheduler_zs` here too
        stop_criterion=stop_criterion,
        n_workers=n_workers,
        sleep_time=0,
        callbacks=[SimulatorCallback()],
        tuner_name="test_zero_shot_transfer"
    )
    tuner.run()


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    metric = "metric_avg_ml_task_performance"
    blackbox_name = "autoencodix_ontix_schc"
    test_task = "schc_RNA_CLIN_reactome"
    
    trial_backend = BlackboxRepositoryBackend(
        blackbox_name=blackbox_name,
        dataset=test_task,
        elapsed_time_attr="metric_elapsed_time",
        surrogate="KNeighborsRegressor",
        surrogate_kwargs={"n_neighbors": 1},
    )
    
    simulate_transfer_benchmark(blackbox_name=blackbox_name, test_task=test_task, trial_backend=trial_backend, metric=metric)

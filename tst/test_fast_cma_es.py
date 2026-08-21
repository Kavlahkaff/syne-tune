import numpy as np

from syne_tune.optimizer.schedulers.searchers.fast_cma_es.fast_cma_es_searcher import (
    FastCMAESSearcher,
)
from syne_tune.config_space import uniform, randint

config_space = {
    "x": uniform(-5.0, 5.0),
    "y": uniform(-5.0, 5.0),
    "z": randint(0, 20),
}


def _assert_config_in_space(config):
    assert -5.0 <= config["x"] <= 5.0
    assert -5.0 <= config["y"] <= 5.0
    assert 0 <= config["z"] <= 20


def test_fast_cma_es_suggest_and_complete():
    searcher = FastCMAESSearcher(config_space, popsize=4, random_seed=0)

    for trial_id in range(20):
        config = searcher.suggest()
        assert config is not None
        _assert_config_in_space(config)
        metric = (config["x"] - 1) ** 2 + (config["y"] + 2) ** 2
        searcher.on_trial_complete(trial_id=trial_id, config=config, metric=metric)


def test_fast_cma_es_points_to_evaluate():
    points_to_evaluate = [{"x": 0.0, "y": 0.0, "z": 5}]
    searcher = FastCMAESSearcher(
        config_space,
        popsize=4,
        random_seed=0,
        points_to_evaluate=points_to_evaluate,
    )

    config = searcher.suggest()
    assert config == points_to_evaluate[0]
    searcher.on_trial_complete(trial_id=0, config=config, metric=1.0)

    for trial_id in range(1, 10):
        config = searcher.suggest()
        _assert_config_in_space(config)
        searcher.on_trial_complete(trial_id=trial_id, config=config, metric=np.random.rand())


def test_fast_cma_es_async_out_of_order_and_errors():
    searcher = FastCMAESSearcher(config_space, popsize=6, random_seed=0)

    n_workers = 3
    pending = {}
    next_trial_id = 0
    num_trials = 40
    rng = np.random.RandomState(0)

    while next_trial_id < num_trials or pending:
        while len(pending) < n_workers and next_trial_id < num_trials:
            config = searcher.suggest()
            _assert_config_in_space(config)
            pending[next_trial_id] = config
            next_trial_id += 1
        if not pending:
            break
        trial_id = rng.choice(list(pending.keys()))
        config = pending.pop(trial_id)
        if rng.rand() < 0.1:
            searcher.on_trial_error(trial_id=trial_id)
        else:
            metric = (config["x"] - 1) ** 2 + (config["y"] + 2) ** 2
            searcher.on_trial_complete(trial_id=trial_id, config=config, metric=metric)

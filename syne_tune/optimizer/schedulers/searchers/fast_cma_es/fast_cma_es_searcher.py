import logging
from typing import Any

import numpy as np

from syne_tune.optimizer.schedulers.searchers.single_objective_searcher import (
    SingleObjectiveBaseSearcher,
)
from syne_tune.optimizer.schedulers.searchers.utils import (
    make_hyperparameter_ranges,
)

try:
    from fcmaes.cmaes import Cmaes
    from scipy.optimize import Bounds
except ImportError as e:
    raise ImportError(
        "FastCMAESSearcher requires the 'fcmaes' package. Install it with "
        "'pip install fcmaes' (or 'pip install syne-tune[extra]')."
    ) from e


logger = logging.getLogger(__name__)


class FastCMAESSearcher(SingleObjectiveBaseSearcher):
    """
    A searcher that uses the ask-tell CMA-ES implementation of
    `fast-cma-es <https://github.com/dietmarwo/fast-cma-es>`__ (package
    ``fcmaes``) to suggest configurations.

    Configurations are encoded into ``[0, 1]``-normalized vectors (via
    :func:`~syne_tune.optimizer.schedulers.searchers.utils.make_hyperparameter_ranges`),
    which ``fcmaes.cmaes.Cmaes`` optimizes. This is an experimental searcher:
    it is best suited to configuration spaces that are (mostly) continuous.

    ``Cmaes`` is a synchronous, population-based optimizer: :meth:`ask`
    returns a whole population of candidates at once, and :meth:`tell`
    expects fitness values for the whole population before the internal
    state (mean, covariance) is updated. Since Syne Tune's searcher API
    hands out and receives results for one trial at a time, and
    asynchronously, this searcher queues up candidates from the current
    population as they are requested via :meth:`suggest`, and only calls
    :meth:`tell` once results for the full population have come back via
    :meth:`on_trial_complete`. If :meth:`suggest` is called with no queued
    candidate left (because the rest of the population is still pending),
    a random configuration is returned instead; its result is not fed back
    into CMA-ES.

    :param config_space: Configuration space for the evaluation function.
    :param points_to_evaluate: A set of initial configurations to be
        evaluated before starting the optimization.
    :param popsize: Population size used by CMA-ES. If ``None``, the default
        population size of ``fcmaes.cmaes.Cmaes`` is used (roughly
        ``4 + 3 * log(dim)``).
    :param random_seed: Seed for initializing random number generators.
    """

    def __init__(
        self,
        config_space: dict[str, Any],
        points_to_evaluate: list[dict] | None = None,
        popsize: int | None = None,
        random_seed: int = None,
    ):
        super(FastCMAESSearcher, self).__init__(
            config_space, points_to_evaluate=points_to_evaluate, random_seed=random_seed
        )
        self._hp_ranges = make_hyperparameter_ranges(config_space)
        bounds = self._hp_ranges.get_ndarray_bounds()
        lower = np.array([lo for lo, _ in bounds])
        upper = np.array([hi for _, hi in bounds])
        self._bounds = Bounds(lower, upper)
        # ``fcmaes.cmaes.Cmaes`` documents ``popsize=None`` as using its own
        # default, but as of fcmaes 2.0.3 that branch is buggy (it divides by
        # the raw ``popsize`` argument before applying the default), so the
        # default population size formula is replicated here instead of
        # relying on ``popsize=None``.
        self._requested_popsize = popsize or (4 + int(3.0 * np.log(len(bounds))))
        self._rng = np.random.default_rng(self.random_seed)

        self._es = self._make_optimizer()
        self._batch_xs = None
        self._batch_ys = dict()
        self._candidate_queue = []
        # Maps internal trial_id (assigned by this searcher) to the index of
        # the candidate within the current CMA-ES batch it corresponds to,
        # or None if it is a random filler candidate not part of the batch.
        self._trial_to_candidate = dict()
        self._next_trial_id = 0
        self._start_new_batch()

    def _make_optimizer(self) -> "Cmaes":
        return Cmaes(
            bounds=self._bounds,
            popsize=self._requested_popsize,
            rg=self._rng,
        )

    def __getstate__(self):
        # ``fcmaes.cmaes.Cmaes`` holds internal ``mmap``-backed buffers that
        # cannot be pickled, which would otherwise break Tuner checkpointing.
        # It is dropped here and rebuilt (with a fresh population) in
        # ``__setstate__``; any in-flight candidates from the batch active at
        # checkpoint time are simply not fed back into CMA-ES once their
        # results arrive after being restored.
        state = self.__dict__.copy()
        del state["_es"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._es = self._make_optimizer()
        self._start_new_batch()
        self._trial_to_candidate = dict()

    def _start_new_batch(self):
        self._batch_xs = self._es.ask()
        self._batch_ys = dict()
        self._candidate_queue = list(range(len(self._batch_xs)))

    def _get_random_config(self) -> dict[str, Any]:
        return self._hp_ranges.from_ndarray(
            self._rng.uniform(size=self._hp_ranges.ndarray_size)
        )

    def suggest(self, **kwargs) -> dict[str, Any] | None:
        config = self._next_points_to_evaluate()
        trial_id = self._next_trial_id
        self._next_trial_id += 1

        if config is not None:
            self._trial_to_candidate[trial_id] = None
            return config

        if self._candidate_queue:
            candidate_index = self._candidate_queue.pop(0)
            self._trial_to_candidate[trial_id] = candidate_index
            config = self._hp_ranges.from_ndarray(self._batch_xs[candidate_index])
        else:
            # Whole current population already handed out, but not all
            # results are back yet: return a filler random config which is
            # not fed back into CMA-ES.
            self._trial_to_candidate[trial_id] = None
            config = self._get_random_config()

        return config

    def on_trial_error(self, trial_id: int):
        candidate_index = self._trial_to_candidate.pop(trial_id, None)
        if candidate_index is None:
            return
        # Treat a failed trial as an infinitely bad observation, so the
        # batch can still complete and CMA-ES steers away from this region.
        self._record_result(candidate_index, np.inf)

    def on_trial_complete(
        self,
        trial_id: int,
        config: dict[str, Any],
        metric: float,
    ):
        candidate_index = self._trial_to_candidate.pop(trial_id, None)
        if candidate_index is None:
            return
        self._record_result(candidate_index, metric)

    def _record_result(self, candidate_index: int, metric: float):
        self._batch_ys[candidate_index] = metric

        if len(self._batch_ys) == len(self._batch_xs):
            ys = np.array([self._batch_ys[i] for i in range(len(self._batch_xs))])
            try:
                self._es.tell(ys, self._batch_xs)
                self._start_new_batch()
            except Exception:
                logger.warning(
                    "fast-cma-es failed to update its internal state, "
                    "restarting the optimizer.",
                    exc_info=True,
                )
                self._es = self._make_optimizer()
                self._start_new_batch()

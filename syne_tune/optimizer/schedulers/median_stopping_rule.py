import logging
from collections import defaultdict
from typing import Any

import numpy as np

from syne_tune.backend.trial_status import Trial
from syne_tune.optimizer.scheduler import TrialScheduler
from syne_tune.optimizer.scheduler import (
    SchedulerDecision,
    TrialSuggestion,
)


logger = logging.getLogger(__name__)


class MedianStoppingRule(TrialScheduler):
    """
    Applies median stopping rule in top of an existing scheduler.

    * If result at time-step ranks less than the cutoff of other results observed
      at this time-step, the trial is interrupted and otherwise, the wrapped
      scheduler is called to make the stopping decision.
    * Suggest decisions are left to the wrapped scheduler.
    * The mode of the wrapped scheduler is used.

    Reference:

        | Google Vizier: A Service for Black-Box Optimization.
        | Golovin et al. 2017.
        | Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge
        | Discovery and Data Mining, August 2017
        | Pages 1487–1495
        | https://dl.acm.org/doi/10.1145/3097983.3098043

    :param scheduler: Scheduler to be called for trial suggestion or when
        median-stopping-rule decision is to continue.
    :param resource_attr: Key in the reported dictionary that accounts for the
        resource (e.g. epoch).
    :param running_average: If ``True``, then uses the running average of
        observation instead of raw observations. Defaults to ``True``
    :param metric: Metric to be considered, defaults to ``scheduler.metric``
    :param grace_time: Median stopping rule is only applied for results whose
        ``resource_attr`` exceeds this amount. Defaults to 1
    :param grace_population: Median stopping rule when at least
        ``grace_population`` have been observed at a resource level. Defaults to 5
    :param rank_cutoff: Results whose quantiles are below this level are
        discarded. Defaults to 0.5 (median)
    :param random_seed: Seed used to initialize the random number generators.
    :param do_minimize: True if we minimize the objective function
    """

    def __init__(
        self,
        scheduler: TrialScheduler,
        resource_attr: str,
        running_average: bool = True,
        metric: str | None = None,
        grace_time: int | None = 1,
        grace_population: int = 5,
        rank_cutoff: float = 0.5,
        random_seed: int = None,
        do_minimize: bool | None = True,
    ):
        super(MedianStoppingRule, self).__init__(random_seed=random_seed)
        if metric is None and hasattr(scheduler, "metric"):
            metric = getattr(scheduler, "metric")
        self.metric = metric
        self.sorted_results = defaultdict(list)
        self.scheduler = scheduler
        self.resource_attr = resource_attr
        self.rank_cutoff = rank_cutoff
        self.grace_time = grace_time
        self.min_samples_required = grace_population
        self.running_average = running_average

        self.metric_multiplier = 1 if do_minimize else -1
        self.do_minimize = do_minimize
        if running_average:
            self.trial_to_results = defaultdict(list)

    def suggest(self) -> TrialSuggestion | None:
        return self.scheduler.suggest()

    def on_trial_result(self, trial: Trial, result: dict) -> str:
        new_metric = result[self.metric] * self.metric_multiplier

        time_step = result[self.resource_attr]

        if self.running_average:
            # gets the running average of current observations
            self.trial_to_results[trial.trial_id].append(new_metric)
            new_metric = np.mean(self.trial_to_results[trial.trial_id])

        # insert new metric in sorted results acquired at this resource
        index = np.searchsorted(self.sorted_results[time_step], new_metric)
        self.sorted_results[time_step] = np.insert(
            self.sorted_results[time_step], index, new_metric
        )
        normalized_rank = index / float(len(self.sorted_results[time_step]))

        if (
            self.grace_condition(time_step=time_step)
            or normalized_rank <= self.rank_cutoff
        ):
            return self.scheduler.on_trial_result(trial=trial, result=result)
        else:
            logger.info(
                f"see new results {new_metric} at time-step {time_step} for trial {trial.trial_id}"
                f" with rank {int(normalized_rank * 100)}%, "
                f"stopping it as it does not rank on the top {int(self.rank_cutoff * 100)}%"
            )
            return SchedulerDecision.STOP

    def grace_condition(self, time_step: float) -> bool:
        """
        :param time_step: Value :code:`result[self.resource_attr]`
        :return: Decide for continue?
        """
        if (
            self.min_samples_required is not None
            and len(self.sorted_results[time_step]) < self.min_samples_required
        ):
            return True
        if self.grace_time is not None and time_step < self.grace_time:
            return True
        return False

    def metadata(self) -> dict[str, Any]:
        """
        :return: Metadata for the scheduler
        """
        metadata = super().metadata()
        metadata_scheduler = self.scheduler.metadata()

        return metadata_scheduler | metadata

    def metric_names(self) -> list[str]:
        return self.metric

    def metric_mode(self) -> str:
        return "min" if self.do_minimize else "max"

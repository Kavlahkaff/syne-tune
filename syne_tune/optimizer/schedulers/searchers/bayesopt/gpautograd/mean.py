from typing import Any
import autograd.numpy as anp
from autograd.tracer import getval

from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.constants import (
    INITIAL_MEAN_VALUE,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.distribution import (
    Normal,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.gluon import Block
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.gluon_blocks_helpers import (
    IdentityScalarEncoding,
    encode_unwrap_parameter,
    register_parameter,
)

__all__ = ["MeanFunction", "ScalarMeanFunction", "ZeroMeanFunction"]


class MeanFunction(Block):
    """
    Mean function, parameterizing a surrogate model together with a kernel function.

    Note: KernelFunction also inherits from this interface.
    """

    def __init__(self, **kwargs):
        Block.__init__(self, **kwargs)

    def param_encoding_pairs(self):
        """
        Returns list of tuples
            (param_internal, encoding)
        over all Gluon parameters maintained here.

        :return: List [(param_internal, encoding)]
        """
        raise NotImplementedError

    def get_params(self) -> dict[str, Any]:
        """
        :return: Dictionary with hyperparameter values
        """
        raise NotImplementedError

    def set_params(self, param_dict: dict[str, Any]):
        """
        :param param_dict: Dictionary with new hyperparameter values
        :return:
        """
        raise NotImplementedError


class ScalarMeanFunction(MeanFunction):
    """
    Mean function defined as a scalar (fitted while optimizing the marginal
    likelihood).

    :param initial_mean_value: A scalar to initialize the value of the mean
    """

    def __init__(self, initial_mean_value=INITIAL_MEAN_VALUE, **kwargs):
        super().__init__(**kwargs)

        # Even though we do not apply specific transformation to the mean value
        # we use an encoding to handle in a consistent way the box constraints
        # of Gluon parameters (like bandwidths or residual noise variance)

        self.encoding = IdentityScalarEncoding(
            init_val=initial_mean_value, regularizer=Normal(0.0, 1.0)
        )

        with self.name_scope():
            self.mean_value_internal = register_parameter(
                self.params, "mean_value", self.encoding
            )

    def forward(self, X):
        """
        Actual computation of the scalar mean function
        We compute mean_value * vector_of_ones, whose dimensions are given by
        the the first column of X

        :param X: input data of size (n,d) for which we want to compute the
            mean (here, only useful to extract the right dimension)
        """
        mean_value = encode_unwrap_parameter(self.mean_value_internal, self.encoding)
        return anp.multiply(anp.ones((getval(X.shape[0]), 1)), mean_value)

    def param_encoding_pairs(self):
        return [(self.mean_value_internal, self.encoding)]

    def get_mean_value(self):
        return encode_unwrap_parameter(self.mean_value_internal, self.encoding)[0]

    def set_mean_value(self, mean_value):
        self.encoding.set(self.mean_value_internal, mean_value)

    def get_params(self) -> dict[str, Any]:
        return {"mean_value": self.get_mean_value()}

    def set_params(self, param_dict: dict[str, Any]):
        self.set_mean_value(param_dict["mean_value"])


class ZeroMeanFunction(MeanFunction):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, X):
        return anp.zeros((getval(X.shape[0]), 1))

    def param_encoding_pairs(self):
        return []

    def get_params(self) -> dict[str, Any]:
        return dict()

    def set_params(self, param_dict: dict[str, Any]):
        pass

import time

import dask
import numpy
from sklearn import gaussian_process


def _dask_tanimito_similarity(a, b):
    """
    Fixme this does not need to use matmul anymore because it's not a single core.
    This can be transitioned to simple row by row dispatching.
    """
    print(f"About to compute tanimoto for array lengths {len(a)} and {len(b)}")
    start = time.time()
    chunk_size = 8_000
    da = dask.array.from_array(a, chunks=chunk_size)
    db = dask.array.from_array(b, chunks=chunk_size)
    aa = dask.array.sum(da, axis=1, keepdims=True)
    bb = dask.array.sum(db, axis=1, keepdims=True)
    ab = dask.array.matmul(da, db.T)
    td = dask.array.true_divide(ab, aa + bb.T - ab)
    td_computed = td.compute()
    print(f"Computed tanimoto similarity in {time.time() - start:.2f}s for array lengths {len(a)} and {len(b)}")
    return td_computed


class TanimotoKernel(gaussian_process.kernels.NormalizedKernelMixin,
                     gaussian_process.kernels.StationaryKernelMixin,
                     gaussian_process.kernels.Kernel):
  """Custom Gaussian process kernel that computes Tanimoto similarity."""

  def __init__(self):
    """Initializer."""

  def __call__(self, X, Y=None, eval_gradient=False):  # pylint: disable=invalid-name
    """Computes the pairwise Tanimoto similarity.

    Args:
      X: Numpy array with shape [batch_size_a, num_features].
      Y: Numpy array with shape [batch_size_b, num_features]. If None, X is
        used.
      eval_gradient: Whether to compute the gradient.

    Returns:
      Numpy array with shape [batch_size_a, batch_size_b].

    Raises:
      NotImplementedError: If eval_gradient is True.
    """
    if eval_gradient:
      raise NotImplementedError
    if Y is None:
      Y = X
    return _dask_tanimito_similarity(X, Y)

class Query:
    @staticmethod
    def greedy(optimizer,
               features,
               n_instances=1):
        """Takes the best instances by inference value sorted in ascending order.

        Args:
          optimizer: BaseLearner. Model to use to score instances.
          features: modALinput. Featurization of the instances to choose from.
          n_instances: Integer. The number of instances to select.

        Returns:
          Indices of the instances chosen.
        """
        return numpy.argpartition(optimizer.predict(features), n_instances)[:n_instances]

class Model:

    @staticmethod
    def get_gaussian_process_estimator(**model_params):
        estimator = gaussian_process.GaussianProcessRegressor(kernel=TanimotoKernel(), **model_params)
        return estimator

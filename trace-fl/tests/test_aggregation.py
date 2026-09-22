import numpy as np
import pytest
from federation.strategy import FedAvgStrategy

def test_fedavg_mathematical_correctness():
    strategy = FedAvgStrategy()
    
    update1 = (10, [np.array([1.0, 2.0]), np.array([3.0, 4.0])])
    update2 = (10, [np.array([3.0, 4.0]), np.array([5.0, 6.0])])
    
    updates = [update1, update2]
    
    aggregated = strategy.aggregate(updates)
    
    np.testing.assert_array_almost_equal(aggregated[0], np.array([2.0, 3.0]))
    np.testing.assert_array_almost_equal(aggregated[1], np.array([4.0, 5.0]))

def test_fedavg_weighted():
    strategy = FedAvgStrategy()
    
    update1 = (10, [np.array([1.0])])
    update2 = (30, [np.array([5.0])])
    
    updates = [update1, update2]
    
    aggregated = strategy.aggregate(updates)
    
    expected = (10 * 1.0 + 30 * 5.0) / 40
    np.testing.assert_array_almost_equal(aggregated[0], np.array([expected]))
    
def test_fedavg_empty():
    strategy = FedAvgStrategy()
    with pytest.raises(ValueError):
        strategy.aggregate([])

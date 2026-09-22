from flwr.common import Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from core.types import NDArrays

def ndarrays_to_flower_parameters(ndarrays: NDArrays) -> Parameters:
    """Converts a list of NumPy arrays into Flower Parameters."""
    return ndarrays_to_parameters(ndarrays)

def flower_parameters_to_ndarrays(parameters: Parameters) -> NDArrays:
    """Converts Flower Parameters into a list of NumPy arrays."""
    return parameters_to_ndarrays(parameters)

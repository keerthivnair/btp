import torch
import numpy as np
from core.model import create_model, get_parameters, set_parameters

def test_model_creation():
    model = create_model()
    assert model is not None

def test_parameter_extraction_and_restoration():
    model1 = create_model()
    model2 = create_model()
    
    # Change model2 weights to be different
    for p in model2.parameters():
        p.data.fill_(1.0)
        
    params1 = get_parameters(model1)
    set_parameters(model2, params1)
    
    params2 = get_parameters(model2)
    
    for p1, p2 in zip(params1, params2):
        np.testing.assert_array_equal(p1, p2)

def test_penultimate_layer():
    model = create_model()
    dummy_input = torch.randn(1, 1, 28, 28)
    penultimate = model.get_penultimate_layer(dummy_input)
    assert penultimate.shape == (1, 128)

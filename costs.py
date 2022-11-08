import torch
import numpy as np
import neuralnetwork
import get_sensitivities
from typing import Callable

class Costs():
    pass


def to_optimise(input: torch.tensor, model: neuralnetwork.NeuralNetwork, cost: Callable, ind_param: int, length_output: int =200):
    somme = 0
    sensitivities = get_sensitivities.sensitivities(input, model, length_output)
    for k in range(length_output):
        somme += sensitivities[:, ind_param + 1]*cost(k, input)
    return somme


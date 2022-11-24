import torch
import neuralnetwork

def save_MDN_model(MDNmodel: neuralnetwork.NeuralNetwork, file_path: str):
    """Saves the parameters of a Mixture Density Network model in a `.pt` file under **file_path**,
    including weights, architecture, number of parameters in input, number of components in the output mixture
    and the type of output mixture.

    Args:
        - **MDNmodel** (neuralnetwork.NeuralNetwork): MDN model to save.
        - **file_path** (str): Path of the file in which to save the parameters.
    """    
    torch.save({'model_state_dict': MDNmodel.state_dict(),
                'n_comps': MDNmodel.n_comps,
                'num_params': MDNmodel.num_params,
                'n_hidden': MDNmodel.n_hidden,
                'mixture': MDNmodel.mixture},
                file_path)

def load_MDN_model(file_path: str) -> neuralnetwork.NeuralNetwork:
    """Loads a Mixture Density Network model from a `.pt` file.

    Args:
        - **file_path** (str): Path of the file in which the parameters of the model are saved.

    Returns:
        - Loaded MDN model.
    """    
    loader = torch.load(file_path)
    MDNmodel = neuralnetwork.NeuralNetwork(loader['n_comps'],
                                        loader['num_params'], 
                                        loader['n_hidden'],
                                        loader['mixture'])
    MDNmodel.load_state_dict(loader['model_state_dict'])
    MDNmodel.eval()
    return MDNmodel
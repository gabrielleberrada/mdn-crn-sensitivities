# DL-based-Control-of-CRNs

- ‘afl_data‘: Data loaded from Nessie github.
- ‘import_afl_data‘: Loads afl data from csv to tensors and separates it into training data, validation data and test data.
- ‘training_afl‘: trains a Neural Network on the afl data.

- ‘simulation‘: CRN and SSA classes which launch simulations for a specified CRN.
- ‘generate_data‘: Generates the data needed for the neural network with SSA simulations. It uses multiprocessing and thus needs the propensity functions defined in a separate file. Returns the data as arrays.
- ‘convert_csv‘: Conversion from array to csv and from csv to array/tensor

- ‘neuralnetwork‘: Creates a neural network model and trains it with the ‘train_NN‘ function.
- ‘training‘: trains a Neural Network and computes the loss for the chosen train, validation and test data.
- ‘grads‘: From a trained model, computes the gradients with respect to the input parameters and calculates the expected value of these gradients.

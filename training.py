import import_afl_data
import neuralnetwork


model = neuralnetwork.NeuralNetwork(import_afl_data.n_comps, import_afl_data.num_params)
trainer_test = neuralnetwork.train_NN(model, import_afl_data.train_data, import_afl_data.valid_data, loss=neuralnetwork.loss_kldivergence, max_rounds=500, lr=0.01, batchsize=64)

print("Training dataset")
print(f"KLD : {neuralnetwork.mean_loss(import_afl_data.X_train, import_afl_data.y_train, model, loss=neuralnetwork.loss_kldivergence)}")
print(f'Hellinger : {neuralnetwork.mean_loss(import_afl_data.X_train, import_afl_data.y_train, model, loss=neuralnetwork.loss_hellinger)}')

print("\nValidation dataset")
print(f"KLD : {neuralnetwork.mean_loss(import_afl_data.X_valid, import_afl_data.y_valid, model, loss=neuralnetwork.loss_kldivergence)}")
print(f'Hellinger : {neuralnetwork.mean_loss(import_afl_data.X_valid, import_afl_data.y_valid, model, loss=neuralnetwork.loss_hellinger)}')

print("\nTest dataset")
print(f"KLD : {neuralnetwork.mean_loss(import_afl_data.X_test, import_afl_data.y_test, model, loss=neuralnetwork.loss_kldivergence)}")
print(f'Hellinger : {neuralnetwork.mean_loss(import_afl_data.X_test, import_afl_data.y_test, model, loss=neuralnetwork.loss_hellinger)}')

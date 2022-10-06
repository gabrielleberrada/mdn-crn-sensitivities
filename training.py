import convert_csv
import neuralnetwork

n_comps = 5
num_params = 1

X_train = convert_csv.csv_to_tensor('birth_data/X_ø_S1_train.csv')
y_train = convert_csv.csv_to_tensor('birth_data/y_ø_S1_train.csv')
X_valid = convert_csv.csv_to_tensor('birth_data/X_ø_S1_valid.csv')
y_valid = convert_csv.csv_to_tensor('birth_data/y_ø_S1_valid.csv')
X_test = convert_csv.csv_to_tensor('birth_data/X_ø_S1_test.csv')
y_test = convert_csv.csv_to_tensor('birth_data/y_ø_S1_test.csv')

data_train = [X_train, y_train]
data_valid = [X_valid, y_valid]
data_test = [X_test, y_test]

model = neuralnetwork.NeuralNetwork(n_comps, num_params)
trainer_test = neuralnetwork.train_NN(model, data_train, data_valid, loss=neuralnetwork.loss_kldivergence, max_rounds=500, lr=0.01, batchsize=64)

print("Training dataset")
print(f"KLD : {neuralnetwork.mean_loss(X_train, y_train, model, loss=neuralnetwork.loss_kldivergence)}")
print(f'Hellinger : {neuralnetwork.mean_loss(X_train, y_train, model, loss=neuralnetwork.loss_hellinger)}')

print("\nValidation dataset")
print(f"KLD : {neuralnetwork.mean_loss(X_valid, y_valid, model, loss=neuralnetwork.loss_kldivergence)}")
print(f'Hellinger : {neuralnetwork.mean_loss(X_valid, y_valid, model, loss=neuralnetwork.loss_hellinger)}')

print("\nTest dataset")
print(f"KLD : {neuralnetwork.mean_loss(X_test, y_test, model, loss=neuralnetwork.loss_kldivergence)}")
print(f'Hellinger : {neuralnetwork.mean_loss(X_test, y_test, model, loss=neuralnetwork.loss_hellinger)}')

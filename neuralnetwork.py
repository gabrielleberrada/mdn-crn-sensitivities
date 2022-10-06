import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from typing import Callable, Tuple


class NeuralNetwork(nn.Module):
    """
    Builds a neural network.
    """
    def __init__(self, n_comps: int, n_params: int, n_hidden: int = 128) -> None:
        """
        Inputs:
        'n_comps': number of components in output.
        'n_params': number of parameters in input, excluding time parameter. 
        'n_hidden': number of neurons in the hidden layer.

        Output:
        The neural network structure.
        """
        super(NeuralNetwork, self).__init__()
        self.n_comps = n_comps
        self.num_params = n_params
        self.hidden_layer = nn.Linear(1 + n_params, n_hidden)
        self.hidden_actf = F.relu
        self.MNBOutputLayer1 = nn.Linear(n_hidden, n_comps)
        self.output_actf1 = nn.Softmax(dim=-1)
        self.MNBOutputLayer2 = nn.Linear(n_hidden, n_comps)
        self.output_actf2 = torch.exp
        self.MNBOutputLayer3 = nn.Linear(n_hidden, n_comps)
        self.output_actf3 = torch.sigmoid
        # initializing with Glorot Uniform method
        nn.init.xavier_uniform_(self.hidden_layer.weight)
        nn.init.xavier_uniform_(self.MNBOutputLayer1.weight)
        nn.init.xavier_uniform_(self.MNBOutputLayer2.weight)
        nn.init.xavier_uniform_(self.MNBOutputLayer3.weight)
        print('Neural Network created.')

    def forward(self, input: torch.tensor) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        """
        Input: the input data to process.

        Output: A tuple of three vectors.
                'layer_ww': mixture weights.
                'layer_rr': count parameters.
                'layer_pp': success probabilities.
        """
        x = torch.log10(input)
        x = self.hidden_layer(x)
        x = self.hidden_actf(x)
        layer_ww = self.MNBOutputLayer1(x)
        layer_rr = self.MNBOutputLayer2(x)
        layer_pp = self.MNBOutputLayer3(x)
        layer_ww = self.output_actf1(layer_ww)
        layer_rr = self.output_actf2(layer_rr)
        layer_pp = self.output_actf3(layer_pp)
        return layer_ww, layer_rr, layer_pp


# Negative Binomial mixtures

def nbpdf(r: torch.tensor, 
        p: torch.tensor, 
        k: torch.tensor, 
        eps:float =1e-5
        ) -> torch.tensor:
    """
    Inputs: A tuple of three vectors.
            'r': count parameters of successes.
            'p': success probabilities.
            'k': points at which to evaluate the pdf.
            'eps': corrective term since a negative binomial cannot be evaluated at p=1.0

    Output: The pdf of a negative binomial distribution NB(r, p) evaluated at k.
    """
    # Here r is the number of successes but pytorch understands it as the number of failures.
    corrected_p = 1 - p
    corrected_p[corrected_p == 1.] -= eps
    distr = torch.distributions.negative_binomial.NegativeBinomial(r, corrected_p)
    return torch.exp(distr.log_prob(k))

def mix_nbpdf(rr: torch.tensor, 
            pp: torch.tensor, 
            ww: torch.tensor, 
            k: torch.tensor
            ) -> torch.tensor:
    """
    Inputs: Parameters of the negative binomial mixtures.
            'rr': count parameters.
            'pp': success probabilities.
            'ww': weights.
            'k': points at which to evaluate the pdf.

    Output: The pdf of a negative binomial mixture distribution evaluated at k.
    """
    ret = torch.mul(ww, nbpdf(rr, pp, k))
    return torch.sum(ret, dim=-1)

def pred_pdf(model: NeuralNetwork, 
            x: torch.tensor, 
            yy: torch.tensor
            ) -> torch.tensor:
    """
    Computes the predicted distribution of the model at input points 'x' and evaluates its pdf at points 'yy'.
    """
    ww, rr, pp = model.forward(x)
    return mix_nbpdf(rr, pp, ww, yy)


## Losses functions

def loss_kldivergence(x: torch.tensor, 
                    y: torch.tensor, 
                    model: NeuralNetwork
                    ) -> float:
    """
    Computes the Kullback-Leibler divergence of 
    the predicted distribution of the model at input points 'x' and of the corresponding outputs.
    """
    y_size = y.size()
    if len(y_size) == 1:
        dim0 = 1
    else:
        dim0 = y_size[0]
    # to provide a set of points at which to evaluate the pdf for each input point.
    mat_k = torch.arange(y_size[-1]).repeat(dim0,model.n_comps,1).permute([2,0,1])
    pred = pred_pdf(model, x, mat_k)
    kl_loss = nn.KLDivLoss(reduction='sum')
    p = pred.permute(1,0)
    p[p==0]=1e-9
    return kl_loss(torch.log(p), y)


def loss_hellinger(x: torch.tensor, 
                y: torch.tensor, 
                model: NeuralNetwork
                ) -> float:
    """
    Computes the Hellinger distance from the predicted distribution points 
    of the model at input points 'x' to the corresponding outputs 'y'.
    """
    y_size = y.size()
    if len(y_size) == 1:
        dim0 = 1
    else:
        dim0 = y_size[0]
    # to provide a set of points at which to evaluate the pdf for each input point.
    mat_k = torch.arange(y_size[-1]).repeat(dim0,model.n_comps,1).permute([2,0,1])    
    pred = pred_pdf(model, x, mat_k)
    # computes the Hellinger distance step by step
    sqrt1 = torch.sqrt(pred.permute(1,0)*y)
    sqrt1 = sqrt1.sum(dim=-1)
    normalization = pred.sum(dim=0) * y.sum(dim=-1)
    sqrt2 = torch.sqrt(normalization)
    elt = torch.div(sqrt1, sqrt2)
    mat = torch.sqrt(torch.add(torch.ones(dim0), -elt))
    return mat.sum()


def mean_loss(X: torch.tensor, 
            y: torch.tensor, 
            model: NeuralNetwork, 
            loss: Callable =loss_kldivergence
            ) -> float:
    """
    Computes the average loss over a batch. 
    
    Input:  'X': vector of inputs
            'y': corresponding vector of outputs.
    """
    ret = loss(X, y, model)
    return ret / len(X)


# Training Neural Net

class TrainArgs:
    """
    Class for training hyperparameters.
    """
    def __init__(self, 
                train_data: Tuple[torch.tensor, torch.tensor], 
                valid_data: Tuple[torch.tensor, torch.tensor], 
                lr: float =0.01, 
                l2_reg: float =0., 
                max_rounds: int =1_000, 
                min_lr: float =0.01/32,
                batchsize: int =100, 
                optimizer: Callable =torch.optim.Adam
                ) -> None:
        self.lr = np.float32(lr)            # Current learning rate
        self.l2_reg = np.float32(l2_reg)    # L2 regularisation weight
        self.max_rounds = max_rounds        # Maximal number of training rounds
        self.min_lr = np.float32(min_lr)    # Minimial learning rate
        if batchsize == 0:
            batchsize = len(train_data[0])
        self.batchsize = batchsize
        self.optimizer = optimizer
        self.train_data = train_data
        self.valid_data = valid_data


class NNTrainer:
    """
    Class for data generated while training the network.
    """
    def __init__(self, 
                args: TrainArgs, 
                model: NeuralNetwork
                ) -> None:
        # divides the dataset into batches.
        train_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(args.train_data[0], args.train_data[1]), 
                                                batch_size=args.batchsize, shuffle=True)
        self.train_loader = train_loader
        self.train_losses = []
        self.valid_losses = []
        self.lr_updates = [0]
        self.args = args
        self.model = model
        self.opt = args.optimizer
        self.iteration = 0
        self.update_losses()

    def update_losses(self) -> None:
        """
        Computes training and validation losses at each iteration.
        """
        train_loss = mean_loss(self.args.train_data[0], self.args.train_data[1], self.model, loss=loss_kldivergence)
        valid_loss = mean_loss(self.args.valid_data[0], self.args.valid_data[1], self.model, loss=loss_kldivergence)
        self.train_losses.append(float(train_loss.detach().numpy()))
        self.valid_losses.append(float(valid_loss.detach().numpy()))

    def should_decrease_lr(self, n_rounds: int =50, tol: float =0.005) -> bool:
        """
        Indicates to reduce the learning rate if at least n_rounds have passed since the last decrease 
        and if the mean validation loss has changed by less than tol% in the last n_rounds rounds.

        Inputs: 'n_rounds': minimal number of rounds betwwen two learning rate updates.
                'tol': tolerance threshold of learning rate variations.

        Outputs: A boolean stating if the learning rate should be decreased.
        """
        losses = self.valid_losses
        round = self.iteration
        if round > (self.lr_updates[-1] + n_rounds):
            return torch.mean(torch.tensor(losses[-n_rounds//2:])).item() > (torch.mean(torch.tensor(losses[-n_rounds:-n_rounds//2])).item() * (1-tol))
        return False

    def __iter__(self):
        return self

    def __next__(self):
        """
        At each iteration, updates the learning rate if needed.
        """
        iter = len(self.train_losses)
        if iter >= self.args.max_rounds:
            raise StopIteration
        if self.should_decrease_lr():
            new_lr = self.args.lr / 2
            if new_lr < self.args.min_lr:
                raise StopIteration
            self.lr_updates.append(iter)
            self.args.lr = new_lr
        return iter+1, self


# Training rounds

def train_round(trainer: NNTrainer, loss: Callable =loss_kldivergence) -> None:
    """
    Performs one training epoch ie train on each datum.
    """
    model = trainer.model
    optimizer = trainer.opt(model.parameters(), lr=trainer.args.lr, weight_decay=trainer.args.l2_reg)
    for x, y in trainer.train_loader:
        model.zero_grad()
        loss_y = mean_loss(x, y, model, loss)
        loss_y.backward()
        optimizer.step()
    trainer.update_losses()


# Training process

def train_NN(model: NeuralNetwork, 
            train_data: Tuple[torch.tensor, torch.tensor], 
            valid_data: Tuple[torch.tensor, torch.tensor], 
            loss: Callable=loss_kldivergence, 
            **kwargs) -> Tuple[list[float], list[float]]:
    """
    Trains the neural network using the given training data and validation data. 
    
    Inputs: 'train_data': tuples '(X_train, y_train)' of data to train the neural network on.
                        'X_train': tensor of input points.
                        'y_train': tensor of corresponding outputs.
            'valid_data': tuples of data to validate the neural network on.
                        'X_valid': tensor of input points.
                        'y_valid': tensor of corresponding outputs.
            'kwargs': other arguments for hyperparameters.

    Outputs:    Training and validation losses for each epoch.
                Prints the learning rate, the train and valid losses at the end of the training.
    """
    args = TrainArgs(train_data, valid_data, **kwargs)
    trainer = NNTrainer(args, model)
    pbar = tqdm(total=args.max_rounds, desc='Training ...', position=0)
    for _ in trainer:
        pbar.update(1)
        # learning rate updates during iteration
        train_round(trainer, loss)
        trainer.iteration += 1
    pbar.close()
    print(f'Learning rate: {trainer.args.lr},\nTrain loss: {trainer.train_losses[-1]},\n Valid loss: {trainer.valid_losses[-1]}')
    return trainer.train_losses, trainer.valid_losses

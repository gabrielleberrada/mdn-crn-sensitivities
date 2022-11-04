import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from typing import Callable, Tuple


class NeuralNetwork(nn.Module):
    """Class to build a Mixture Density Network.

    Args:
        - **n_comps** (int): Number of components of the mixture in output.
        - **n_params** (int): Number of CRN parameters in input, excluding time parameter. 
        - **n_hidden** (int, optional): Number of neurons in the hidden layer. Defaults to 128.
        - **mixture** (str, optional): Type of mixture to compute. Defaults to 'NB' for a Negative Binomial mixture. Can also be 'Poisson' for a Poisson mixture.
        - **print_info** (bool, optional): If True, prints 'Mixture Density Network created' once the network is built. Defaults to True.
    """                
    def __init__(self, 
                n_comps: int, 
                n_params: int, 
                n_hidden: int =128,
                mixture: str ='NB',
                print_info: bool=True):
        super(NeuralNetwork, self).__init__()
        self.mixture = mixture
        self.n_comps = n_comps
        self.num_params = n_params
        self.hidden_layer = nn.Linear(1 + n_params, n_hidden)
        self.hidden_actf = F.relu
        # initializing with Glorot Uniform method
        nn.init.xavier_uniform_(self.hidden_layer.weight)
        # weights layer
        self.MNBOutputLayer1 = nn.Linear(n_hidden, n_comps)
        self.output_actf1 = nn.Softmax(dim=-1)
        nn.init.xavier_uniform_(self.MNBOutputLayer1.weight)
        # parameters layer
        self.MNBOutputLayer2 = nn.Linear(n_hidden, n_comps)
        self.output_actf2 = F.relu
        nn.init.xavier_uniform_(self.MNBOutputLayer2.weight)
        if mixture == 'NB':
            # success probabilities layer
            self.MNBOutputLayer3 = nn.Linear(n_hidden, n_comps)
            self.output_actf3 = torch.sigmoid
            nn.init.xavier_uniform_(self.MNBOutputLayer3.weight)
        if print_info:
            print('Mixture Density Network created.')

    def forward(self, input: torch.tensor) -> Tuple[torch.tensor]:
        """Runs the forward function of the Mixture Density Network.

        Args:
            - **input** (torch.tensor[float]): The input data to process.

        Returns:
            - A tuple of three tensors.
                - **layer_ww**: Mixture weights.

                If Negative Binomial Mixture:
                    - **layer_rr**: Count parameters.
                    - **layer_pp**: Success probabilities.

                If Poisson Mixture:
                    - **layer_rr**: Rate parameters.
        """        
        x = torch.log10(input)
        x = self.hidden_layer(x)
        x = self.hidden_actf(x)
        layer_ww = self.MNBOutputLayer1(x)
        layer_ww = self.output_actf1(layer_ww)
        layer_rr = self.MNBOutputLayer2(x)
        layer_rr = self.output_actf2(layer_rr)
        if self.mixture == 'NB':
            layer_pp = self.MNBOutputLayer3(x)
            layer_pp = self.output_actf3(layer_pp)
            return layer_ww, layer_rr, layer_pp
        return layer_ww, layer_rr


# Negative Binomial mixtures

def distr_pdf(params: Tuple,
        k: torch.tensor, 
        mixture: str,
        eps:float =1e-5,
        ) -> torch.tensor:
    """Computes the pdf of the components of the mixture.

    Args:
        - **params** (Tuple): Parameters needed to define the probability distribution.
        - **k** (torch.tensor): Points at which to evaluate the pdf.
        - **mixture** (str): Name of the chosen distribution for the mixture.
        - **eps** (float, optional): Corrective term since a negative binomial cannot be evaluated at p=1.0. Defaults to 1e-5.

    Returns:
        - The pdf of the distribution evaluated at k.
    """    
    # NB distribution
    if mixture == 'NB':
        # Here r is the number of successes but pytorch understands it as the number of failures.
        r, p = params
        corrected_p = 1 - p
        corrected_p[corrected_p == 1.] -= eps
        distr = torch.distributions.negative_binomial.NegativeBinomial(r, corrected_p)
        prob = distr.log_prob(k)
    # Poisson distribution
    if mixture == 'Poisson':
        # Here p is the rate of the process.
        r = params[0]
        corrected_r = r.clone()
        corrected_r[corrected_r < eps] += eps
        distr = torch.distributions.poisson.Poisson(corrected_r)
        prob = distr.log_prob(k)
        # to avoid errors with very small probabilities
        prob[prob < -10] = -10
    return torch.exp(prob)


def mix_pdf(model: NeuralNetwork, 
            x: torch.tensor, 
            yy: torch.tensor
            ) -> torch.tensor:
    """Computes the predicted distribution of the Neural Network at input points **x** and evaluates its pdf at points **yy**.
    
    Parameters of the distribution mixture:
        - **ww**: Weights of the mixture.
        - **params**: Parameters to define the distribution.

    A distribution mixture evaluated at point k is given by:

    .. math::

        q(k) = \sum_i w_i Distr(n, params_i)

    Args:
        - **model** (NeuralNetwork): Mixture Density Network to use.
        - **x** (torch.tensor): Input points, [t, param1, ...].
        - **yy** (torch.tensor): Points at which to evaluate the pdf.

    Returns:
        - The pdf of a mixture of distributions evaluated at k.
    """
    output = model.forward(x)
    ww, params = output[0], output[1:]
    ret = torch.mul(ww, distr_pdf(params, yy, mixture=model.mixture))
    return torch.sum(ret, dim=-1)


## Losses functions

def loss_kldivergence(x: torch.tensor, 
                    y: torch.tensor, 
                    model: NeuralNetwork
                    ) -> float:
    r"""Computes the Kullback-Leibler divergence of the predicted distribution of the Neural Network at input points **x** and of the corresponding outputs.

   For tensors of the same shape :math:`y_{pred}` and :math:`y_{target}`:

    .. math::

        KL(y_{pred}, y_{target}) = y_{target} \log \bigl(\frac{y_{target}}{y_{pred}}\bigl)

    Args:
        - **x** (torch.tensor): Vector of inputs.
        - **y** (torch.tensor): Corresponding vector of outputs :math:`y_{target}`.
        - **model** (NeuralNetwork): Mixture Density Network to use.

    Returns:
        - The Kullback-Leibler divergence value.
    """
    y_size = y.size()
    if len(y_size) == 1:
        dim0 = 1
    else:
        dim0 = y_size[0]
    # to provide a set of points at which to evaluate the pdf for each input point.
    mat_k = torch.arange(y_size[-1]).repeat(dim0,model.n_comps,1).permute([2,0,1])
    pred = mix_pdf(model, x, mat_k)
    p = pred.permute(1,0)
    # log gradient is not defined near 0.
    p[p<1e-10]=1e-10
    y[y<1e-10]=1e-10
    kl_loss = nn.KLDivLoss(reduction='sum')
    return kl_loss(torch.log(p), y)

def loss_hellinger(x: torch.tensor, 
                y: torch.tensor, 
                model: NeuralNetwork
                ) -> float:
    """Computes the Hellinger distance of the predicted distribution of the Neural Network at input points **x** and of the corresponding outputs.

    For tensors of the same shape :math:`y_{pred}` and :math:`y_{target}`:

    .. math::

        H^2(y_{pred}, y_{target}) = 1 - \sum_i \sqrt(y_{pred, i})\sqrt(y_{target, i})

    Args:
        - **x** (torch.tensor): Vector of inputs.
        - **y** (torch.tensor): Corresponding vector of outputs.
        - **model** (NeuralNetwork): Mixture Density Network to use.

    Returns:
        - :math:`H(y_{pred}, y)` (float): Hellinger distance between the predicted distribution of the Neural Network at input points **x** and the expected output **y**.
    """
    y_size = y.size()
    if len(y_size) == 1:
        dim0 = 1
    else:
        dim0 = y_size[0]
    # to provide a set of points at which to evaluate the pdf for each input point.
    mat_k = torch.arange(y_size[-1]).repeat(dim0,model.n_comps,1).permute([2,0,1])    
    pred = mix_pdf(model, x, mat_k)
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
    """Computes the average loss over a batch.

    Args:
        - **X** (torch.tensor): Vector of inputs.
        - **y** (torch.tensor): Corresponding vector of outputs.
        - **model** (NeuralNetwork): Mixture Density Network to use.
        - **loss** (Callable, optional): Loss to use. Defaults to loss_kldivergence.

    Returns:
        - Average loss between the predicted distribution of the Neural Network at input points **x** and the expected output **y**.
    """
    ret = loss(X, y, model)
    return ret / len(X)


# Training Mixture Density Network
class NNTrainer:
    """Class to train the Mixture Density Network.

    Hyperparameters and parameters are saved in the dictionary `args`.

    Args:
        - **model** (NeuralNetwork): Mixture Density Network to use.
        - **train_data** (Tuple[torch.tensor, torch.tensor]): Training dataset.
        - **valid_data** (Tuple[torch.tensor, torch.tensor]): Validation dataset.
        - :math:`l_r` (float, optional): Current learning rate. Defaults to 0.01.
        - **l2_reg** (float, optional): L2-regularization term. Defaults to 0.
        - **max_rounds** (int, optional): Maximal number of training rounds. Defaults to 1_000.
        - **batchsize** (int, optional): Number of elements in a batch. Defaults to 100.
        - **optimizer** (Callable, optional): Optimizer to use. Defaults to torch.optim.Adam.
        - **patience** (int, optional): Number of events needed without improvement to stop the training. Defaults to 20.
        - **delta** (float, optional): Minimum value difference between the lowest and the second lowest results. Defaults to 1e-5.
    """
    def __init__(self,
                model: NeuralNetwork,
                train_data: Tuple[torch.tensor, torch.tensor], 
                valid_data: Tuple[torch.tensor, torch.tensor], 
                lr: float =0.01, 
                l2_reg: float =0.,
                max_rounds: int =1_000,
                batchsize: int =100, 
                optimizer: Callable =torch.optim.Adam,
                patience: int =25,
                delta: float =1e-5
                ):               
        self.args = {
            'train_data': train_data,
            'valid_data': valid_data,
            'lr': lr,
            'l2_reg': l2_reg,
            'max_rounds': max_rounds,
            'batchsize': batchsize,
            'optimizer': optimizer
        }
        # divides the dataset into shuffled batches.
        self.train_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(self.args['train_data'][0], 
                                                                                        self.args['train_data'][1]), 
                                                    batch_size=self.args['batchsize'], shuffle=True)
        self.train_losses = []
        self.valid_losses = []
        self.lr_updates = [0]
        self.model = model
        self.opt = self.args['optimizer'](model.parameters(), lr=self.args['lr'], weight_decay=self.args['l2_reg'])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, self.args['max_rounds'])
        self.iteration = 0
        self.patience=patience
        self.delta=delta
        self.update_losses()

    def update_losses(self):
        """
        Computes training and validation losses at each iteration and saves them.
        """
        train_loss = mean_loss(self.args['train_data'][0], self.args['train_data'][1], self.model, loss=loss_kldivergence)
        valid_loss = mean_loss(self.args['valid_data'][0], self.args['valid_data'][1], self.model, loss=loss_kldivergence)
        self.train_losses.append(float(train_loss.detach().numpy()))
        self.valid_losses.append(float(valid_loss.detach().numpy()))

    def early_stopping(self) -> bool:
        """Computes early stopping when the Neural Network does not improve to avoid overfitting.
        If **patience** number of rounds pass without getting the validation loss at least closer than **delta** to the best validation loss achieved,
        stops the training.

        Returns:
            - A boolean stating if the training should be stopped.
        """    
        if len(self.valid_losses) < self.patience:
            return False
        losses = np.array(self.valid_losses[-self.patience:])
        losses[0] += self.delta
        if np.argmin(losses) == 0:
            return True
        return False

    def __iter__(self):
        return self

    def __next__(self):
        iter = len(self.train_losses)
        if iter >= self.args['max_rounds'] or self.early_stopping():
            raise StopIteration
        return iter+1, self


# Training rounds

def train_round(trainer: NNTrainer, loss: Callable =loss_kldivergence):
    """Performs one training epoch.

    Args:
        - **trainer** (NNTrainer): Training structure.
        - **loss** (Callable, optional): Loss to use for optimization. Defaults to loss_kldivergence.
    """
    model = trainer.model
    optimizer = trainer.opt
    scheduler = trainer.scheduler
    for x, y in trainer.train_loader:
        model.zero_grad()
        loss_y = mean_loss(x, y, model, loss)
        loss_y.backward()
        # gradient clip to tackle exploding gradient.
        nn.utils.clip_grad_norm_(model.parameters(), 10.)
        optimizer.step()
    # updates learning rate
    scheduler.step()
    trainer.args['lr'] = scheduler.get_last_lr()[0]
    trainer.lr_updates.append(scheduler.get_last_lr()[0])
    trainer.update_losses()


# Training process

def train_NN(model: NeuralNetwork, 
            train_data: Tuple[torch.tensor, torch.tensor], 
            valid_data: Tuple[torch.tensor, torch.tensor], 
            loss: Callable =loss_kldivergence,
            print_results: bool =True, 
            print_info: bool =True,
            **kwargs) -> Tuple[list[float], list[float]]:
    """Trains the Neural Network.

    Args:
        - **model** (NeuralNetwork): Mixture Density Network to use.
        - **train_data** (Tuple[torch.tensor, torch.tensor]): tuple **(X_train, y_train)** of data to train the Neural Network on.

                        - **X_train**: Tensor of input data.
                        - **y_train**: Tensor of expected outputs.
        - **valid_data** (Tuple[torch.tensor, torch.tensor]): tuple **(X_valid, y_valid)** of data to validate the Neural Network on.

                        - **X_valid**: Tensor of input data.
                        - **y_valid**: Tensor of expected outputs.
        - **loss** (Callable, optional): Loss to use for optimization. Defaults to loss_kldivergence.
        - **print_results** (bool, optional): If True, prints the final results (learning rate, train and valid losses at the end of the training). Defaults to True.
        - **print_info** (bool, optional): If True, prints a progress bar. Defaults to True.

    Returns:
        - Training and validation losses for each epoch.
    """            
    trainer = NNTrainer(model, train_data, valid_data, **kwargs) # not sure
    if print_info:
        pbar = tqdm(total=trainer.args['max_rounds'], desc='Training ...', position=0)
    for _ in trainer:
        if print_info:
            pbar.update(1)
        train_round(trainer, loss)
        trainer.iteration += 1
    if print_info:
        pbar.close()
    if print_results:
        print(f"Learning rate: {trainer.args['lr']},\nTrain loss: {trainer.train_losses[-1]},\n Valid loss: {trainer.valid_losses[-1]}")
    return trainer.train_losses, trainer.valid_losses


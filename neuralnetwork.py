import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from typing import Callable, Tuple


class NeuralNetwork(nn.Module):
    r"""Class to build a Mixture Density Network.
    Based on :cite:`sukys2022nessie`.

    Args:
        - **n_comps** (int): Number of components of the output mixture.
        - **n_params** (int): Number of parameters in input :math:`M_{\text{tot}}`, excluding the time parameter.
        - **n_hidden** (int, optional): Number of neurons in the hidden layer. Defaults to :math:`128`.
        - **mixture** (str, optional): Type of mixture to compute. Defaults to `NB` for a Negative Binomial mixture.
          Can also be `Poisson` for a Poisson mixture.
        - **print_info** (bool, optional): If True, prints 'Mixture Density Network created' once the Neural Network is built. 
          Defaults to True.
    """                
    def __init__(self, 
                n_comps: int, 
                n_params: int, 
                n_hidden: int =128,
                mixture: str ='NB',
                print_info: bool =True):
        super(NeuralNetwork, self).__init__()
        self.mixture = mixture
        self.n_comps = n_comps
        self.num_params = n_params
        self.n_hidden = n_hidden
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
        r"""Runs the forward function of the Mixture Density Network.

        Args:
            - **input** (torch.tensor): The input parameters to predict: 
              :math:`[t, \theta_1, ..., \theta_{M_{\theta}}, \xi_1^1, \xi_1^2, ..., \xi_1^{M_{\xi}},\xi_2^1, ..., \xi^{M_{\xi}}_L]`.

        Returns:
            - A tuple of three tensors.
                - **layer_ww**: Mixture weights.

                For a Negative Binomial Mixture:
                    - **layer_rr**: Count parameters.
                    - **layer_pp**: Success probabilities.

                For a Poisson Mixture:
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

def distr_pdf(params: Tuple[torch.tensor],
        k: torch.tensor, 
        mixture: str,
        eps: float =1e-5,
        ) -> torch.tensor:
    """Computes the probability density function (pdf) of the components of the mixture.

    Args:
        - **params** (Tuple[torch.tensor]): Parameters needed to define the probability distribution.
        - **k** (torch.tensor): Points at which to evaluate the pdf.
        - **mixture** (str): Name of the chosen distribution for the mixture.
        - **eps** (float, optional): Corrective term since a Negative Binomial cannot be evaluated at :math:`p=1.0`.
          Defaults to :math:`10^{-5}`.

    Returns:
        - The pdf of the distribution evaluated at **k**.
    """    
    # NB distribution
    if mixture == 'NB':
        # Here r is the number of successes but pytorch requires a number of failures.
        r, p = params
        corrected_p = 1 - p
        # to avoid errors with very small probabilities
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
    r"""Computes the predicted distribution of the Neural Network at input points **x** and evaluates its pdf at points **yy**.
    
    Parameters of the distribution mixture:
        - **ww**: Weights of the mixture.
        - **params**: Parameters to define the distribution.

    A distribution mixture evaluated at point **k** is given by:

    .. math::

        q(k) = \sum_i w_i \text{Distr}(k, \text{params}_i)

    Args:
        - **model** (NeuralNetwork): Mixture Density Network model.
        - **x** (torch.tensor): Input points, 
          :math:`[t, \theta_1, ..., \theta_{M_{\theta}}, \xi_1^1, \xi_1^2, ..., \xi_1^{M_{\xi}}, ..., \xi_L^{M_{\xi}}]`.
        - **yy** (torch.tensor): Points at which to evaluate the pdf.

    Returns:
        - The pdf of a mixture of distributions evaluated at **k**.
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
    r"""Computes the Kullback-Leibler divergence from the predicted distribution at input points **x** to the expected output **y**.

   For tensors of the same shape :math:`\hat{y}` and :math:`y`:

    .. math::

        KL(y, \hat{y}) = y \log \bigl(\frac{y}{\hat{y}}\bigl)

    Args:
        - **x** (torch.tensor): Vector of inputs.
        - **y** (torch.tensor): Expected vector of outputs :math:`y`.
        - **model** (NeuralNetwork): Mixture Density Network model.

    Returns:
        - :math:`KL(y, \hat{y})`: Kullback-Leibler divergence between the predicted fistribution of the Neural Network at input
          points **x** and the expected output **y**.
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
    # correction: gradient of log is not defined near 0
    p[p<1e-10]=1e-10
    y[y<1e-10]=1e-10
    kl_loss = nn.KLDivLoss(reduction='sum')
    # in pytorch, 1st argument is the prediction and 2nd one is the expected result
    # the logarithm is computed for y but not p
    return kl_loss(torch.log(p), y)

def loss_hellinger(x: torch.tensor, 
                y: torch.tensor, 
                model: NeuralNetwork
                ) -> float:
    """Computes the Hellinger distance from the predicted distribution at input points **x** to the expected output **y**.

    For tensors of the same shape :math:`\hat{y}` and :math:`y`:

    .. math::

        H(y, \hat{y}) =\sqrt{1 - \sum_i \sqrt{\hat{y}_iy_i}}

    Args:
        - **x** (torch.tensor): Vector of inputs.
        - **y** (torch.tensor): Expected vector of outputs.
        - **model** (NeuralNetwork): Mixture Density Network model.

    Returns:
        - :math:`H(y, \hat{y})`: Hellinger distance between the predicted distributions of the Neural Network at input points **X**
          and the expected outputs **y**.
    """
    y_size = y.size()
    if len(y_size) == 1:
        dim0 = 1
    else:
        dim0 = y_size[0]
    # to provide a set of points at which to evaluate the pdf for each input point.
    mat_k = torch.arange(y_size[-1]).repeat(dim0,model.n_comps,1).permute([2,0,1])    
    pred = mix_pdf(model, x, mat_k)
    # computing the Hellinger distance step by step
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
        - **y** (torch.tensor): Expected vector of outputs.
        - **model** (NeuralNetwork): Mixture Density Network model.
        - **loss** (Callable, optional): Chosen loss. Defaults to `loss_kldivergence`.

    Returns:
        - Average loss between the predicted distribution of the Neural Network at input points **x** and the expected output **y**.
    """
    ret = loss(X, y, model)
    return ret / len(X)


# Training Mixture Density Network

class NNTrainer:
    """Class to train the Mixture Density Network model.

    Hyperparameters and parameters are saved in the dictionary `args`.

    Args:
        - **model** (NeuralNetwork): Mixture Density Network model.
        - **train_data** (Tuple[torch.tensor, torch.tensor]): Training dataset.
        - **valid_data** (Tuple[torch.tensor, torch.tensor]): Validation dataset.
        - :math:`l_r` (float, optional): Initial learning rate. Defaults to :math:`0.005`.
        - **l2_reg** (float, optional): L2-regularisation term. Defaults to :math:`0`.
        - **max_rounds** (int, optional): Maximal number of epochs. Defaults to :math:`700`.
        - **batchsize** (int, optional): Number of elements in a batch. Defaults to :math:`64`.
        - **optimiser** (Callable, optional): Chosen optimiser. Defaults to torch.optim.Adam.
        - **add_early_stopping** (Tuple[bool, int, float], optional): Defaults to (False, :math:`50`, :math:`10^{-6}`).  
        
            - (bool): If True, uses the early stopping regularisation. Defaults to False.            
            - **patience** (int): Patience level :math:`n_p`. 
              At epoch :math:`n`, the :math:`(n-n_p)` -th epoch is compared pairwise with that 
              of the last :math:`n_p` epochs. Defaults to :math:`50`.
            - **delta** (float): Tolerance threshold :math:`\delta`. Training is stopped if the decrease between 
              the elements of one of those pairs is lower than :math:`\delta`. Defaults to :math:`10^{-6}`.          
    """
    def __init__(self,
                model: NeuralNetwork,
                train_data: Tuple[torch.tensor, torch.tensor], 
                valid_data: Tuple[torch.tensor, torch.tensor], 
                lr: float =0.005, 
                l2_reg: float =0.,
                max_rounds: int =700,
                batchsize: int =64, 
                optimiser: Callable =torch.optim.Adam,
                add_early_stopping: Tuple[bool, int, float] =(False, None, 1e-6)
                ):               
        self.args = {
            'train_data': train_data,
            'valid_data': valid_data,
            'lr': lr,
            'l2_reg': l2_reg,
            'max_rounds': max_rounds,
            'batchsize': batchsize,
            'optimiser': optimiser,
            'patience': add_early_stopping[1],
            'delta': add_early_stopping[2]
        }
        # divides the dataset into shuffled batches.
        self.train_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(self.args['train_data'][0], 
                                                                                        self.args['train_data'][1]), 
                                                        batch_size=self.args['batchsize'], 
                                                        shuffle=True) # to reshuffle after each epoch
        self.train_losses = []
        self.valid_losses = []
        self.lr_updates = [0]
        self.model = model
        self.opt = self.args['optimiser'](model.parameters(), lr=self.args['lr'], weight_decay=self.args['l2_reg'])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, self.args['max_rounds'])
        self.iteration = 0
        self.add_early_stopping = add_early_stopping[0]
        self.update_losses()

    def update_losses(self):
        """
        Computes training and validation losses at each iteration and stores them.
        """
        train_loss = mean_loss(self.args['train_data'][0], self.args['train_data'][1], self.model, loss=loss_kldivergence)
        valid_loss = mean_loss(self.args['valid_data'][0], self.args['valid_data'][1], self.model, loss=loss_kldivergence)
        self.train_losses.append(float(train_loss.detach().numpy()))
        self.valid_losses.append(float(valid_loss.detach().numpy()))

    def early_stopping(self) -> bool:
        """Computes early stopping regularisation to avoid overfitting.
        If :math:`n_p` number of rounds pass without improving the validation loss by at least :math:`\delta`,
        stops the training.

        Only called if `add_early_stopping` is True.

        Returns:
            - A boolean stating if the training should be stopped.
        """
        if len(self.valid_losses) < self.args['patience']:
            return False
        losses = np.array(self.valid_losses[-self.args['patience']:])
        losses[0] -= self.args['delta']
        if np.argmin(losses) == 0:
            return True
        return False

    def __iter__(self):
        return self

    def __next__(self):
        iter = len(self.train_losses)
        if iter >= self.args['max_rounds']:
            raise StopIteration
        # early stopping regularisation
        if self.add_early_stopping and self.early_stopping():
            raise StopIteration
        return iter+1, self


# Training rounds

def train_round(trainer: NNTrainer, loss: Callable =loss_kldivergence):
    """Performs one training epoch.

    Args:
        - **trainer** (NNTrainer): Training structure.
        - **loss** (Callable, optional): Chosen loss for optimisation. Defaults to `loss_kldivergence`.
    """
    model = trainer.model
    optimiser = trainer.opt
    scheduler = trainer.scheduler
    for x, y in trainer.train_loader:
        model.zero_grad()
        loss_y = mean_loss(x, y, model, loss)
        loss_y.backward()
        # clipping gradients to tackle exploding gradient.
        nn.utils.clip_grad_norm_(model.parameters(), 10.)
        optimiser.step()
    # updating learning rate
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
            **kwargs) -> Tuple[list]:
    """Trains the Neural Network.

    Args:
        - **model** (NeuralNetwork): Mixture Density Network model to train.
        - **train_data** (Tuple[torch.tensor, torch.tensor]): Training dataset.

                        - **X_train**: Tensor of input data.
                        - **y_train**: Tensor of expected outputs.
        - **valid_data** (Tuple[torch.tensor, torch.tensor]): Validation dataset. Only used for early stopping.

                        - **X_valid**: Tensor of input data.
                        - **y_valid**: Tensor of expected outputs.
        - **loss** (Callable, optional): Chosen loss for optimisation. Defaults to `loss_kldivergence`.
        - **print_results** (bool, optional): If True, prints the final results
          (final learning rate, train and valid losses). Defaults to True.
        - **print_info** (bool, optional): If True, prints a progress bar. Defaults to True.

    Returns:
        - Training and validation losses for each epoch.
    """
    trainer = NNTrainer(model, train_data, valid_data, **kwargs)
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


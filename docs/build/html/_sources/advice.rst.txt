Some advice on the implementation of the approach
=================================================

Here are some general observations regarding Mixture Density Networks.

Which mixture should I choose?
------------------------------

It is recommended to choose a Negative Binomial mixture as it has performed better in all studied examples.
It tends to fit various distributions better and also avoids overfitting the data.


Do I need to use regularisation methods while training?
-------------------------------------------------------

As long as the model does not overfit, no need need to use regularisation terms. In most of the studied examples, 
models showed a good generalization ability.

How can I detect overfitting?
-----------------------------

Overfitting can be indicate by a higher final loss for the validation dataset compared to the training dataset.

To confirm or refute overfitting, you can plot the training and validation losses produced by the training function. 
If the validation loss curve is increasing after a certain number of iterations, it suggests that the model is overfitting.

You can plot the losses using the following code:

.. code-block:: python

    import neuralnetwork
    import matplotlib.pyplot as plt
    
    model = neuralnetwork.NeuralNetwork(n_comps=N_COMPS, n_params=NUM_PARAMS, n_hidden=N_HIDDEN, mixture=mixture)
    train_losses, valid_losses = neuralnetwork.train_NN(model, train_data, valid_data, loss=neuralnetwork.loss_kldivergence, max_rounds=N_ITER, lr=LR, batchsize=BATCHSIZE)

    plt.plot(train_losses, '+', label='Training loss')
    plt.plot(valid_losses, 'x', label='Validation loss')
    plt.legend()
    plt.show()

How do I deal with overfitting?
-------------------------------

There are various methods to improve a Neural Network generalisation ability.

The implemented code already allows to add a :math:`\ell_2`-regularisation term or to use an early stopping method.

Early stopping works as follows: At epoch :math:`n`, it compares pairwise the loss of the :math:`(n-n_p)`-th epoch with that of the last :math:`n_p` epochs. 
The Neural Network is considered to have improved if the decrease between the elements of one of those pairs is greater than :math:`\delta`. 
If the Neural Network has not improved, the descent is stopped.

How many simulations should I run to create a dataset?
------------------------------------------------------

It is recommended to run multiple simulations for each data to deduce the appropriate probability mass function.
At least :math:`10^4` simulations should be run to ensure that the mass function is accurate enough.

The size of the training dataset depends on the complexity of the mass function to predict. For simple systems,
a dateset size of :math:`10^3` has been used. For more complex systems, a larger dataset may be necessary. For instance, 
the dataset size used for the controlled Toggle switch Reaction Network was around :math:`4 \cdot 10^4`.

What hyperparameters should I choose for the Projected Gradient Descent Algorithm?
----------------------------------------------------------------------------------

Hyperparameters are crucial for the Projected Gradient Descent algorithm to perform well. 

The step size :math:`\gamma` is usually taken between :math:`10^{-3}` and :math:`1`. 
The maximal number of iterations parameter guarantees the termination of the program. It should be set high enough to never be reached.

The hyperparameters can be tuned by testing multiple combinations and choosing the most efficient one, just as for the Neural Network hyperparameters:.







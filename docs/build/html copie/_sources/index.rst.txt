.. DL for Fisher Information documentation master file, created by
   sphinx-quickstart on Mon Oct 31 14:18:19 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. To publish the documentation: https://sphinx-rtd-tutorial.readthedocs.io/en/latest/read-the-docs.html

Efficient Fisher Information Computation and Policy Search in Sampled Stochastic Chemical Reaction Networks through Deep Learning
==================================================================================================================================

This project introduces a Deep-Learning based method to compute the probability mass function and the sensitivity of the likelihood of Chemical Reaction Networks in an accurate and fast manner.
It first demonstrates the performance of the method by computing the Fisher Information, then uses the Neural Network model to perform fast Optimal Control experiments.



State of the art
----------------

The Fisher Information
^^^^^^^^^^^^^^^^^^^^^^

Computing the Fisher Information currently relies on approximating an expectation by a finite sum, which involves the probability mass
function and the sensitivity of the likelihood.



.. We work on a jump process :math:`(X_t)_{t \in \mathbb{R}_+}` and introduce its mass function :math:`p_t^\theta = p(\cdot, t, \theta) : \mathbb{N}^N \rightarrow \mathbb{R}`.

.. The Fisher Information  of the jump process :math:`(X_t)_{t \in \mathbb{R}_+}` is defined as:

.. .. math::

..    \mathcal{I}_t^\theta = E_{\theta}\Big[\big(\nabla_\theta \log p(X_t;t,\theta)\big)^\top \big(\nabla_\theta \log p(X_t;t,\theta)\big)\Big]


.. Whenever the state-space of :math:`X_t` is finite, each element :math:`[\mathcal{I}_t^\theta]_{ij}` can be rewritten as:

.. .. math::

..    [\mathcal{I}_t^\theta]_{ij} = [S_t^\theta]^\top \text{Diag}\bigg(\frac{1}{p_t^\theta}\bigg)S_t^\theta

.. where :math:`p : \mathbb{N}^N \times \mathbb{R}_+ \times \mathbb{R}_+^M \rightarrow \mathbb{R}_+` is the likelihood and 
.. :math:`S_t^\theta \in \mathcal{M}_{N_\max, M}(\mathbb{R})` the sensitivity of the likelihood.

Details on the formulas and notations can be found in the section :ref:`Background on the Fisher Information<Background on the Fisher information>`.

.. We consider Chemical Reaction Networks as Markov pure jump processes. 

.. Introducing :math:`(X_t)_{t \in \mathbb{R}_+}` the distribution of the jump process, its :ref:`Fisher Information<Background on the Fisher information>` 
.. :math:`\mathcal{I}_t^\theta \in \mathcal{M}_{M,M}(\mathbb{R})` is defined as:

.. .. math::

..    \mathcal{I}_t^\theta = E_{p_t^\theta}\Big[\big(\nabla_\theta \log p(X_t;\theta)\big)^\top \big(\nabla_\theta \log p(X_t;\theta)\big)\Big]

.. If the state-space of :math:`X_t` is finite, each element :math:`[\mathcal{I}_t^\theta]_{ij}` can be rewritten as:

.. .. math::

..    [\mathcal{I}_t^\theta]_{ij} = \sum_{\ell = 1}^{N_{\max}} \frac{1}{p_\ell(t,\theta)} [S_t^\theta]_{\ell i} [S_t^\theta]_{\ell j}

.. Where:

.. - :math:`\theta` is the chosen set of parameters for this Chemical Reaction Network.
.. - :math:`p_t^\theta` is :math:`X_t` probability mass function.
.. - :math:`S_t^\theta` is its stoichiometry matrix.
.. - :math:`N_{\max}` is its cardinality.
.. - We write the ordered elements of the projected state-space as :math:`\{ x_t^1, x_t^2, ..., x_t^{N_{\max}} \}`.

Current methods
""""""""""""""""

The current state-of-the-art is based on two main methods:

- Monte Carlo estimations associated to finite differences.
- The Finite State Projection (FSP) process and its Chemical Master Equation :cite:`fox2019fspfim`.

Limitations
""""""""""""

Both these methods show major limitations:

#. Both methods are affected by the curse of dimensionality when the number of species and the cardinality of their state-space increase.
#. Monte-Carlo method estimates is both highly inefficient and biased.
#. The accuracy of the estimated mass functions is not guaranteed to transfer to the sensitivity of the likelihood, and no error bound has currently been provided in this case.
#. The Finite State Projection method suffers from the degradation of the estimates with time.
#. The evaluation of the Fisher Information for different parameter configurations implies repeatedly solving a usually high-dimensional Ordinary
   Differential Equation.

New approach
------------

To overcome these problems, we propose to use a Mixture Density Network model (as presented in :cite:`sukys2022nessie`) to predict the probability mass function and the sensitivity of the likelihood.
The Fisher Information will follow easily.

Details on the architecture of the Mixture Density Network can be found in the section :ref:`Background on the Architecture of the Mixture Density Network<Background on the Architecture of the Mixture Density Network>`.

Benefits over existing methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Neural Networks offer an alternative way of computing the probability mass functions of Chemical Reaction Networks. Their introduction has been
motivated by their universal approximation property, their generalisation ability and their compositional structure.

The computation of composite functions gradient has been made straightforward to implement and efficient to perform thanks to the development
of dedicated software and hardware. Taking advantages of these strengths, Mixture Density Networks showed accurate estimations of the marginal distributions
for a range of Chemical Reaction Networks :cite:`sukys2022nessie`.

Evaluating the Fisher Information for even a large set of parameters comes at only a marginal cost and is almost instantaneous.
Their generalisation property allows Mixture Density Networks to accurately predict distributions even for times and parameters out of the training range.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Policy search
-------------

Having demonstrated their efficiency and accuracy in estimating probability mass functions and sensitivities of the likelihood, 
the Mixture Density Networks can be used to accurately perform fast policy search.

The aim is to control Chemical Reaction Networks at discrete time points. To do so, we perform a Projected Gradient Descent to optimise a performance index and find the optimal control parameters.

Our experiments demonstrate the speed and accuracy of the Deep-Learning method to do Stochastic control of Chemical Reaction Networks. Using Deep-Learning proved to be 
:math:`31` to :math:`824` times faster than using the Finite State Projection method on the studied examples. Details on the experiments can be found in 
the article :cite:`policysearch2023` as well as the notebooks presented for each example in the `Github repository <https://github.com/gabrielleberrada/DL_based_Control_of_CRNs>`_.

Details on the formulas, the Projected Gradient Descent algorithm and the notations can be found in the section 
:ref:`Background on the Stochastic control of Chemical Reaction Networks<Background on the Stochastic control of Chemical Reaction Networks>`.


User guide
----------

The Python implementation has been made available online on the `GitHub repository DL_based_Control_of_CRNs repository <https://github.com/gabrielleberrada/DL_based_Control_of_CRNs>`_.
The complete method is described and illustrated in the following sections.

.. toctree::
   workflow
   demos
   advice

For information on a specific function, class or method:

.. toctree::
   usage

For more information on the theoretical results:

.. toctree::
   neuralnetwork
   control
   math

.. toctree::
   sources

* :ref:`genindex`
* :ref:`modindex`
Background on the Stochastic control of Chemical Reaction Networks
==================================================================

Our goal is to find control parameters such that the abundance of the species of interst reaches target values at specified time points.

Using the notations from :cite:`ctsb2023`, let us introduce a number :math:`L` of time windows and a piecewise-constant control input :math:`\xi`, taking the value :math:`\xi_l` 
over the time window :math:`l`.

Let us now introduce the Optimal Control performance index :math:`C_{\xi}^J` defined as follows:

.. math::

    C_{\xi}^J = \sum_{l=1}^L w_l \big(E_{\theta, \xi}[X_{t_l}^J]-h_l\big)^2

where :math:`J` is a set of cardinality :math:`1`, :math:`w \in \mathbb{R}_+^L` is the weight vector and :math:`h \in \mathbb{R}_+^L` is the target vector.

Our aim then is to find :math:`\xi^*`, such that:

.. math::

    \xi^* = \text{argmin}_{\xi \in \mathcal{H}} C_{\xi}^J

where :math:`\mathcal{H} = \prod\limits_{k=1}^{M_{\xi}\times L} [a_k, b_k]` with :math:`a, b \in \mathbb{R}^{M_{\xi} \times L}` is a contraint set.

This constrained optimisation program is solved using Projected Gradient Descent, which projects optimisation parameters obtained by Gradient Descent onto
the constraint hypercube :math:`\mathcal{H}`.

.. figure:: ../../docs/source/images/PGD_algo.jpeg
    :class: border
    :width: 700
    :align: center

Note that the optimisation algorithm is carried out using :math:`3` hyperparameters: the maximal number of iterations :math:`n_{\text{iter}}`, the step size :math:`\gamma` and the threshold level :math:`\xi`.

Specific details can be found in the article :cite:`policysearch2023`. 
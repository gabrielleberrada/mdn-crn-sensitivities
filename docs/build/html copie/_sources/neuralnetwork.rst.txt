Background on the Architecture of the Mixture Density Network
==============================================================

Mixture Density Networks for uncontrolled Reaction Networks
-----------------------------------------------------------

The general architecture of the Mixture Density Network is based on :cite:`sukys2022nessie`.

In the following, to simplify the notations, we will omit the writing of parameters :math:`\xi` and assume that :math:`M = M_{\theta}`.

.. figure:: ../../docs/source/images/neuralnetwork_architecture.jpeg
    :class: border
    :width: 600
    :align: center

    Mixture Density Network architecture

The Mixture Density Network takes as inputs:

- The time :math:`t \in \mathbb{R}_+^*`.
- The propensity parameters :math:`\theta \in (\mathbb{R}_+^*)^M` for the chosen Chemical Reaction Network.

It returns as outputs:

- The mixture weights :math:`w \in [0,1]^K`.
- The count parameters :math:`r \in R_+^K`.
- The success probabilities :math:`q \in [0,1]^K`.

These outputs can be used to define the mass function :math:`\hat{p}(.;t,\theta)`:

.. math::
    
    \hat{p}(x;t,\theta) = \sum_{k=1}^K w_k(t,\theta) p_{\text{NB}}\big(x;r_k(t,\theta), q_k(t,\theta)\big)

Where :math:`K` is the number of components in the distribution mixture and :math:`p_{\text{NB}}` is the probability mass function 
of a Negative Binomial distribution :math:`\mathcal{NB}(r,q)`, defined as follows:

.. math::
    \forall k \in \mathbb{N}, p_{\text{NB}}(k;r,q) = \binom{k+r-1}{r-1}(1-q)^kq^r

Mixture Density Networks for controlled Reaction Networks
---------------------------------------------------------

In the general case, each control parameter value is an additional input to the Mixture Density Network.

The Mixture Density Network now takes as inputs:

- The time :math:`t \in \mathbb{R}_+^*`.
- The propensity parameters of the uncontrolled reactions :math:`\theta \in (\mathbb{R}_+^*)^{M_{\theta}}`.
- The propensity parameters of the controlled reactions :math:`(\xi_1^1, ..., \xi_1^{M_{\xi}}, \xi_2^1, ..., \xi_L^{M_{\xi}}) \in (\mathbb{R}_+^*)^{M_\xi \times L}`.

The outputs remain the same:

- The mixture weights :math:`w \in [0,1]^K`.
- The count parameters :math:`r \in R_+^K`.
- The success probabilities :math:`q \in [0,1]^K`.

The rest of the architecture does not change and the Mixture Density Network should be able to fit with the time windows.


Background on the Fisher Information
====================================

Chemical Reaction Networks
--------------------------

We consider Chemical Reaction Networks as Markov pure jump processes (see :cite:`anderson2015stochastic`). 

We introduce :math:`(X_t)_{t \in \mathbb{R}_+}` the distribution of the jump process. 

In the same setting as introduced in :cite:`policysearch2023`, :math:`X_t` has a probability mass function :math:`p_t^\theta`, whose evaluation at the point :math:`x \in \mathbb{N}^N` 
we write :math:`p(x; t, \theta)`. We also introduce the likelihood :math:`p : \mathbb{N}^N \times \mathbb{R}_+ \times \mathbb{R}_+^M \rightarrow \mathbb{R}_+`.

Approximation of the mass function and the sensitivity of the likelihood
------------------------------------------------------------------------

If the state-space of :math:`X_t` is finite, it can be projected into a finite subset of :math:`\mathbb{N}` whose cardinality will be written :math:`N_{\max} \in \mathbb{N}^*`. 
We will write the ordered elements of the projected state-space as :math:`\big\{x^1_t, x^2_t, ..., x^{N_{\max}}_t \big\}`.

For ease of notation, we introduce :

.. math::
   
   p_i(t,\theta)=p(x^i;t,\theta) \quad \forall i \in [\![1, N_{\max}]\!]

We then define the sensitivity :math:`S^\theta_t \in \mathcal{M}_{N_{\max},M}(\mathbb{R})` of the likelihood of :math:`X_t` as:

.. math::

    \big[S_t^\theta\big]_{ij} = \frac{\partial p_i}{\partial \theta_j}(t, \theta)

Fisher Information of sampled Chemical Reaction Networks
--------------------------------------------------------

Let us define the Fisher Information :math:`\mathcal{I}_t^\theta \in \mathcal{M}_{M, M}(\mathbb{R})` of :math:`X_t` as :cite:`fox2019fspfim`:

.. math::

   \mathcal{I}_t^\theta
   = E_{\theta}\Big[ \big(\nabla_{\theta} \log p(X_t; t,\theta)\big)^\top\big(\nabla_{\theta} \log p(X_t; t,\theta)\big)\Big]


If the state-space of :math:`X_t` is finite`, :math:`\mathcal{I}_t^\theta` can be rewritten as:

.. math::

   \mathcal{I}_t^\theta = \big[ S_t^\theta \big]^\top \text{Diag}\Big(\frac{1}{p_t^\theta}\Big)S_t^\theta

Specific details can be found in the article :cite:`policysearch2023`.

Both the Deep Learning and Finite State Projection methods use this formula to estimate the Fisher Information for sampled Chemical Reaction Networks.


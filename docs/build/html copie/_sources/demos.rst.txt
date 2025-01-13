Examples of Chemical Reaction Networks
======================================

Selected Chemical Reaction Networks are used below to demonstrate how the Fisher Information of Chemical Reaction Networks 
can be computed using Mixture Density Networks.

The Chemical Reaction Networks selected in this study are those presented in the article :cite:`fox2019fspfim`.

Production and Degradation Chemical Reaction Network
----------------------------------------------------

- Reactions: 

.. math:: \emptyset \xrightarrow{\theta_1} S

.. math:: S \xrightarrow{\theta_2} \emptyset

- Propensity functions: 

.. math:: 

    \lambda_1(x;\theta) = \theta_1 \quad \lambda_2(x;\theta) = \theta_2 x

- Stoichiometry matrix: 

.. math:: \begin{pmatrix} 1 \\ -1 \end{pmatrix}

Find code and figures in the notebook `CRN_production_degradation.ipynb <https://github.com/gabrielleberrada/DL_based_Control_of_CRNs/blob/main/CRN_production_degradation/CRN_production_degradation.ipynb>`_.

.. figure:: ../../docs/source/images/CRN2_fig1.jpeg
    :class: border
    :width: 400
    :align: center

    Sensitivities of the likelihood with respect to parameter :math:`\theta_1` for the Production and Degradation Chemical Reaction Network.
    Parameters :math:`\theta_1 = 1.5665, \theta_2 = 0.1997`. From left to right, top to bootom, plots correspond to times :math:`t \in [5, 10, 15, 20]`.

.. figure:: ../../docs/source/images/CRN2_bars_fig3.jpg
    :class: border
    :width: 400
    :align: center

    Element :math:`[I_t^\theta]_{11}` of the Fisher Information as a function of time for the Production and Degradation Chemical Reaction Network.
    Parameters :math:`\theta_1 = 1.5665, \theta_2 = 0.1997`. Time :math:`t=30` is outside of the training range.

Controlled Production and Degradation Chemical Reaction Network
---------------------------------------------------------------

- Reactions: 

.. math:: \emptyset \xrightarrow{\theta} S

.. math:: S \xrightarrow{\xi} \emptyset

- Propensity functions: 

.. math:: 
    
    \lambda_1(x;\theta,\xi) = \theta \quad \lambda_2(x;\theta,\xi) = \xi x

- Stoichiometry matrix: 

.. math:: \begin{pmatrix} 1 \\ -1 \end{pmatrix}

Find code and figures in the notebook `CRN_controlled_production_degradation.ipynb <https://github.com/gabrielleberrada/DL_based_Control_of_CRNs/blob/main/CRN_controlled_production_degradation/CRN_controlled_production_degradation.ipynb>`_.


.. figure:: ../../docs/source/images/CRN2_expgrad1_bars.jpg
    :class: border
    :width: 400
    :align: center

    Derivative of the expectation with respect to :math:`\xi_1` for the Production and Degradation Chemical Reaction Network.
    Parameters :math:`\theta = 1.6869, \xi_1 = 1.8234, \xi_2 = 0.3082, \xi_3 = 0.1011, \xi_4 = 1.7771`. Times :math:`t \in [5, 10, 15, 20]`.


.. figure:: ../../docs/source/images/CRN2_exp_results.jpeg
    :class: border
    :width: 400
    :align: center

    Evolution of the abundance of :math:`S` for the Production and Degradation Chemical Reaction Network using the parameters :math:`\xi^*` obtained by policy search.
    Find details of the experiment in the article :cite:`policysearch2023`.

Bursting Gene Chemical Reaction Network
---------------------------------------

- Reactions: 

.. math:: \emptyset \xrightarrow{\theta_1} S_1
.. math:: S_1 \xrightarrow{\theta_2} \emptyset
.. math:: S_1 \xrightarrow{\theta_3} S_1 + S_2
.. math:: S_2 \xrightarrow{\theta_4} \emptyset

- Propensity functions: 

.. math:: 

    \begin{array}{lr}
    \lambda_1(x;\theta) = \theta_1(1-x_1) & \lambda_2(x;\theta) = \theta_2x_1 \\
    \lambda_3(x;\theta) = \theta_3 x_1 & \lambda_4(x;\theta) = \theta_4 x_2
    \end{array}
    

- Stoichiometry matrix: 

.. math:: \begin{pmatrix} 1 & 0 \\ -1 & 0 \\ 0 & 1 \\ 0 & -1 \end{pmatrix}

Find code and figures in the notebook `CRN_bursting_gene.ipynb <https://github.com/gabrielleberrada/DL_based_Control_of_CRNs/blob/main/CRN_bursting_gene/CRN_bursting_gene.ipynb>`_.

.. figure:: ../../docs/source/images/CRN4_SI_fig9.jpg
    :class: border
    :width: 400
    :align: center

    Sensitivites of the likelihood with respect to parameter :math:`\theta_3` for the Bursting gene Chemical Reaction Network.
    Parameters :math:`\theta_1 = 0.6409, \theta_2 = 2.0359, \theta_3 = 0.2688, \theta_4 = 0.0368`.
    From left to right, top to bootom, plots correspond to times :math:`t \in [5, 10, 15, 20]`.

.. figure:: ../../docs/source/images/CRN4_bars_fig7.jpg
    :class: border
    :width: 400
    :align: center

    Element :math:`[I_t^\theta]_{22}` of the Fisher Information as a function of time for the Bursting gene Chemical Reaction Network.
    Parameters :math:`\theta_1 = 0.6409, \theta_2 = 2.0359, \theta_3 = 0.2688, \theta_4 = 0.0368`.

Controlled Bursting gene Chemical Reaction Network
--------------------------------------------------

- Reactions: 

.. math:: \emptyset \xrightarrow{\theta_1} S_1
.. math:: S_1 \xrightarrow{\xi} \emptyset
.. math:: S_1 \xrightarrow{\theta_2} S_1 + S_2
.. math:: S_2 \xrightarrow{\theta_3} \emptyset

- Propensity functions: 

.. math:: \lambda_1(x;\theta) = \theta_1 (1-x_1)
.. math:: \lambda_2(x;\theta) = \xi x_1
.. math:: \lambda_3(x;\theta) = \theta_2 x_1
.. math:: \lambda_4(x;\theta) = \theta_3 x_2

- Stoichiometry matrix: 

.. math:: \begin{pmatrix} 1 & 0 \\ -1 & 0 \\ 0 & 1 \\ 0 & -1 \end{pmatrix}

Find code and figures in the notebook `CRN_controlled_bursting_gene.ipynb <https://github.com/gabrielleberrada/DL_based_Control_of_CRNs/blob/main/CRN_controlled_bursting_gene/CRN_controlled_bursting_gene.ipynb>`_.

.. figure:: ../../docs/source/images/CRN4_expgrad1_bars.jpg
    :class: border
    :width: 400
    :align: center

    Gradient of the expectation with respect to :math:`\xi_1` for the controlled Bursting gene Chemical Reaction Network.
    Parameters :math:`\theta_1 = 0.4622, \theta_2 = 4.9699, \theta_3 = 0.7501, \xi_1 = 0.4677, \xi_2 = 1.3301, \xi_3 = 2.2814, \xi_4 = 0.0594`.


.. figure:: ../../docs/source/images/CRN4_exp_results.jpg
    :class: border
    :width: 400
    :align: center

    Evolution of the abundance of :math:`S_2` for the controlled Bursting gene Chemical Reaction Network using the parameters :math:`\xi^*` obtained by policy search.
    Find details of the experiment in the article :cite:`policysearch2023`.


Controlled Toggle switch Chemical Reaction Network
--------------------------------------------------

- Reactions: 

.. math:: 
    \emptyset \xrightarrow{\lambda_1} S_1 \quad \emptyset \xrightarrow{\lambda_2} S_2
.. math:: 
    S_1 \xrightarrow{\lambda_3} \emptyset \quad S_2 \xrightarrow{\lambda_4} \emptyset

- Propensities

.. math:: 
        \begin{array}{ll}
    \lambda_1(x;t,\theta,\xi) = \theta_1 + \frac{\theta_3}{1+\theta_6(x_t^2)^{\theta_8}} & \lambda_2(x;t,\theta,\xi) = \theta_2 + \frac{\theta_4}{1+\theta_5(x_t^1)^{\theta_7}} \\
    \lambda_3(x;t,\theta,\xi) = \theta_9 x_t^1 & \lambda_4(x;t,\theta,\xi) = \xi x_t^2
    \end{array}

- Stoichiometry matrix

.. math:: \begin{pmatrix} 1 & 0 \\ -1 & 0 \\ 0 & 1 \\ 0 & -1 \end{pmatrix}

Find code and figures in the notebook `CRN_controlled_toggle_switch.ipynb <https://github.com/gabrielleberrada/DL_based_Control_of_CRNs/blob/main/CRN_controlled_toggle_switch/CRN_controlled_toggle_switch.ipynb>`_.

.. figure:: ../../docs/source/images/CRN6_FI_4.jpg
    :class: border
    :width: 400
    :align: center

    Element :math:`[I_t^\theta]_{22}` of the Fisher Information as a function of time for the Toggle switch Chemical Reaction Network.
    Parameters :math:`\theta_1 = 0.7455, \theta_2 = 0.3351, \theta_3 = 0.0078, \theta_4 = 0.4656, \theta_5 = 0.0193, \theta_6 = 0.2696`, 
    :math:`\theta_7 = 2.5266, \theta_8 = 0.4108, \theta_9 = 0.68880, \xi_1 = 0.9276, \xi_2 = 0.2132, \xi_3 = 0.8062, \xi_4 = 0.3897`.

.. figure:: ../../docs/source/images/CRN6_gradient2.jpg
    :class: border
    :width: 400
    :align: center

    Gradient of the expectation with respect to :math:`\xi_1` for the controlled Toggle switch Chemical Reaction Network.
    Parameters :math:`\theta_1 = 0.7455, \theta_2 = 0.3351, \theta_3 = 0.0078, \theta_4 = 0.4656, \theta_5 = 0.0193, \theta_6 = 0.2696`, 
    :math:`\theta_7 = 2.5266, \theta_8 = 0.4108, \theta_9 = 0.68880, \xi_1 = 0.9276, \xi_2 = 0.2132, \xi_3 = 0.8062, \xi_4 = 0.3897`.


.. figure:: ../../docs/source/images/CRN6_exp_results.jpg
    :class: border
    :width: 400
    :align: center

    Evolution of the abundance of :math:`S_2` for the controlled Toggle switch Chemical Reaction Network using the parameters :math:`\xi^*` obtained by policy search.
    Find details of the experiment in the article :cite:`policysearch2023`.

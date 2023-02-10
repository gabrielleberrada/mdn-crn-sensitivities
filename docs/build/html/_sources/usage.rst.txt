API Documentation
=================

Requirements
-------------

This framework requires the installation of:

- Concurrent.futures
- Matplotlib
- NumPy
- PyTorch
- SciPy

See the file `requirements.txt <https://github.com/gabrielleberrada/DL_based_Control_of_CRNs/blob/main/requirements.txt>`_ for more details on the package requirements.

This section outlines all the classes and functions used in the workflow.

Some useful notations are gathered in the following table:

+-------------------------------------------------------------------------------------------------+-------------------------------+
| Denomination                                                                                    | Notation                      |
+-------------------------------------------------------------------------------------------------+-------------------------------+
| Fisher Information at time :math:`t`                                                            | :math:`I_t^{\theta,\xi}`      |
+-------------------------------------------------------------------------------------------------+-------------------------------+
| Number of control parameters of the propensity functions :math:`\xi`                            | :math:`M_{\xi}`               |
+-------------------------------------------------------------------------------------------------+-------------------------------+
| Number of elements in the case of a finite state-space                                          | :math:`N_{\max}`              |
+-------------------------------------------------------------------------------------------------+-------------------------------+
| Number of fixed parameters of the propensity functions :math:`\theta`                           | :math:`M_{\theta}`            |
+-------------------------------------------------------------------------------------------------+-------------------------------+
| Number of reactions                                                                             | :math:`M`                     |
+-------------------------------------------------------------------------------------------------+-------------------------------+
| Number of species                                                                               | :math:`N``                    |
+-------------------------------------------------------------------------------------------------+-------------------------------+
| Number of time windows                                                                          | :math:`L`                     |
+-------------------------------------------------------------------------------------------------+-------------------------------+
| Probability mass function evaluated at the :math:`\ell`-th element of the enumerated state-space| :math:`p_\ell(t, \theta, \xi)`|
+-------------------------------------------------------------------------------------------------+-------------------------------+
| Sensitivity of the likelihood matrix                                                            | :math:`S_t^{\theta,\xi}`      |
+-------------------------------------------------------------------------------------------------+-------------------------------+
| Total number of parameters of the propensity functions :math:`M_{\theta} + M_{\xi}\times L`     | :math:`M_{\text{tot}}`        |
+-------------------------------------------------------------------------------------------------+-------------------------------+



Simulating Chemical Reaction Networks using Stochastic Simulation Algorithms
-----------------------------------------------------------------------------

.. autoclass:: simulation.CRN
    :members:

.. autoclass:: simulation.StochasticSimulation
    :members:

Multiple simulations
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: generate_data.CRN_Dataset
    :members:

.. autoclass:: generate_data.CRN_Simulations
    :members:

Saving data
^^^^^^^^^^^

.. autofunction:: convert_csv.array_to_csv

.. autofunction:: convert_csv.csv_to_array

.. autofunction:: convert_csv.csv_to_tensor

.. autofunction:: generate_csv.generate_csv_datasets

.. autofunction:: generate_csv.generate_csv_simulations


Building and training a Mixture Density Network
------------------------------------------------

.. autoclass:: neuralnetwork.NeuralNetwork
    :members:

.. autofunction:: neuralnetwork.distr_pdf

.. autofunction:: neuralnetwork.mix_pdf

.. autofunction:: neuralnetwork.loss_kldivergence

.. autofunction:: neuralnetwork.loss_hellinger

.. autofunction:: neuralnetwork.mean_loss

.. autoclass:: neuralnetwork.NNTrainer
    :members:

.. autofunction:: neuralnetwork.train_round

.. autofunction:: neuralnetwork.train_NN

The example file ``training.py`` trains a Mixture Density Network model on the Production and Degradation Chemical Reaction Network using these functions.


Computing probability mass functions and the sensitivity of the likelihood
--------------------------------------------------------------------------


.. autofunction:: get_sensitivities.probabilities

.. autofunction:: get_sensitivities.sensitivities


Computing the Fisher Information
--------------------------------

.. autofunction:: get_fi.fisher_information_t

.. autofunction:: get_fi.fisher_information


Computing the expectation and its gradient
------------------------------------------

.. autofunction:: get_sensitivities.expected_val

.. autofunction:: get_sensitivities.gradient_expected_val

Performing a Projected Gradient Descent
---------------------------------------

.. autoclass:: projected_gradient_descent.ProjectedGradientDescent
    :members:

.. autoclass:: projected_gradient_descent.ProjectedGradientDescent_CRN
    :members:

.. autoclass:: projected_gradient_descent.ProjectedGradientDescent_MDN
    :members:

.. autofunction:: projected_gradient_descent.control_method

.. autofunction:: training_pgd.pgdMDN

The example file ``training_pgd.py`` runs the Projected Gradient Descent Algorithm using these functions.

Tuning hyperparameters
-----------------------

.. autofunction:: hyperparameters_test.test_comb

.. autofunction:: hyperparameters_tuning.test_multiple_combs

The example file ``tuning.py`` tunes the hyperparameters for the Production and Degradation Chemical Reaction Network using these functions.


Comparing with the Finite State Projection method
-------------------------------------------------

.. autoclass:: fsp.StateSpaceEnumeration
    :members:

.. autoclass:: fsp.SensitivitiesDerivation
    :members:

.. autoclass:: projected_gradient_descent.ProjectedGradientDescent_FSP
    :members:

.. autofunction:: training_pgd.pgdFSP


Plotting distributions, Fisher Information and expectations
------------------------------------------------------------

.. autofunction:: plot.plot_model

.. autofunction:: plot.multiple_plots


Plotting Fisher Information results
-----------------------------------

.. autofunction:: plot.fi_table

.. autofunction:: plot.fi_barplots

Plotting expectation results
----------------------------

.. autofunction:: plot.expect_val_table

.. autofunction:: plot.expect_val_barplots



Saving and loading a Mixture Density Network
--------------------------------------------

.. autofunction:: save_load_MDN.save_MDN_model

.. autofunction:: save_load_MDN.load_MDN_model

# Recensements des données

## CRN1 - Birth reaction network

Chemical reaction: $ø \rightarrow S$

File names:

Parameters used: 

`datasets = {'train': 1280, 'valid': 128, 'test': 640}`
`DATA_LENGTH = sum(datasets.values())`

`stoich_mat = np.array([1]).reshape(1,1)`
`crn = simulation.CRN(stoichiometric_mat=stoich_mat, propensities=np.array([propensities.lambda1]), n_params=1)`
`dataset = generate_data.CRN_Dataset(crn=crn, sampling_times=np.array([5, 10, 15, 20]))`
`X, y = dataset.generate_data(data_length=DATA_LENGTH, n_trajectories=10**4, sobol_length=2.)`

Time to generate data: 1h14min16s

Time to train neural network: 3min 15

Losses :

- Training dataset
    - KLD : 0.0014171284856274724
    - Hellinger : 0.018422860652208328

- Validation dataset
    - KLD : 0.0013740354916080832
    - Hellinger : 0.018310008570551872

- Test dataset
    - KLD : 0.0014052807819098234
    - Hellinger : 0.01843174174427986




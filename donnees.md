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

Negative Binomials mixture.

Time to generate data: 1h14min16s

Time to train neural network: 3min 15s

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

## CRN2 - Birth/death reaction network

Chemical reaction: $ø \xrightleftharpoons S$

Time to generate data: 2h07min50s

For 3 training sets and 3 validation sets: 4h26min35s

Time to train neural network: 2min44s

Sobol length : $[2., 1.]$.

Initial state: $[0.]$.

Negative Binomials mixture

Losses :

- Training dataset
    - KLD : 0.0019413335248827934
    - Hellinger : 0.015614477917551994

- Validation dataset
    - KLD : 0.0020220025908201933
    - Hellinger : 0.015908943489193916

- Test dataset
    - KLD : 0.0019058112520724535
    - Hellinger : 0.01564248465001583

### Model 1

2min26 d'entraînement.

Training dataset
KLD : 0.0006802836433053017
Hellinger : 0.012737615965306759

Validation dataset
KLD : 0.0006928137736395001
Hellinger : 0.012826388701796532

Test dataset
KLD : 0.0006804928416386247
Hellinger : 0.012790861539542675

### Model 2

2m32

Training dataset
KLD : 0.0006629383424296975
Hellinger : 0.012499656528234482

Validation dataset
KLD : 0.0006592536810785532
Hellinger : 0.012404898181557655

Test dataset
KLD : 0.0006640579667873681
Hellinger : 0.012532493099570274

### Model 3

2min30

Training dataset
KLD : 0.0008214800618588924
Hellinger : 0.013866422697901726

Validation dataset
KLD : 0.0008749182452447712
Hellinger : 0.013479584828019142

Test dataset
KLD : 0.000801032641902566
Hellinger : 0.013783343136310577

## CRN3 - Birth reaction network

Chemical reaction: $S \rightarrow 2S$

Initial state: $\[5]$.

Sobol length : $[0.1]$.

Poisson distribution

Time to generate data (4096 samples) for r=5: 6h 49min 5s (pause 20 min)







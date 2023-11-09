# Trcmed

## Nonparametric modeling of the composite effect of multiple nutrients on blood glucose dynamics.

Treatment-response curves (TRCs) capture the dynamic behavior of temporal physiological signals when subjected to various sparse, noisy, irregularly-sampled treatments.
We introduce several probabilistic treatment-response modeling approaches, which help to:
* Make personalized predictions under varying treatment setups,
* Evaluate the shape of each treatment type function and its personal effect on the overall physiological quantity.

This approach enhances the accuracy of the modeling process and equips clinicians with valuable insights into the impact of specific treatment types on the overall physiological outcome.

## Installation

To start the work - just create the project from the repository.

To install all the needed packages run:
```console
conda install --file requirements.txt
```

## Usage
#### Dataset
We have:
- simple synthetic dataset with data for 3 individuals - /trcmed-kit/data/synthetic/processed_data/.
- real dataset with data for 12 individuals - /trcmed-kit/data/real/processed_data/.

#### Parametric models
There are two parametric models - P-Resp and P-IDR (written in R and inferred with Stan).

The Stan code - /trcmed-kit/src/models/parametric/CHOSEN_MODEL/SimpleModelHier.stan

To compile and run the .stan file - /trcmed-kit/src/run/parametric/CHOSEN_MODEL/SimpleModelHier.R.

To work with .R models it is better to use HPC cluster and run
```shell
/trcmed-kit/src/run/parametric/run_parametric_triton.sh
```
Inside the shell script the desired parametric model address can be chosen, as well as the output directories.

After the sampling has been done and the results stored in the respective folder, the python code to plot results:
```
python /trcmed-kit/src/plot/parametric/CHOSEN_MODEL.py
```
Results (plots and metrics) are stored in /trcmed-kit/data/{real or synthetic}/results_data/parametric/CHOSEN_MODEL/ folder.


#### Nonparametric models
There are three nonparametric models - GP-Resp, GP-LFM and GP-Conv.

Run nonparametric model:
```
python /trcmed-kit/src/run/non_parametric/CHOSEN_MODEL/run_hierarchical.py
```
Results (plots and metrics) are stored in /trcmed-kit/data/{real or synthetic}/results_data/non_parametric/CHOSEN_MODEL/ folder.
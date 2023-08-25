# Trcmed

## "Nonparametric modeling of the composite effect of multiple nutrients on blood glucose dynamics."

Treatment-response curves (TRCs) capture the dynamic behavior of temporal physiological signals when subjected to various sparse, noisy, irregularly-sampled treatments.
We introduce several probabilistic treatment-response modeling approaches, which help to:
* Make personalized predictions under varying treatment setups,
* Evaluate the shape of each treatment type function and its personal effect on the overall physiological quantity.

This approach enhances the accuracy of the modeling process and equips clinicians with valuable insights into the impact of specific treatment types on the overall physiological outcome.

## Installation

To work with the project just create the project from the repository.

To install all the needed packages:
```console
conda install --file requirements.txt
```

## Usage
#### Dataset
For that submission stage, we use the simple synthetic dataset with data for three individuals. It is stored in /trcmed_submission/data/processed_data/ folder and is divided into train and test parts.
As the data is artificial, it may not have some mechanistic/biological tendencies underneath as opposed to the real one.
#### Parametric models
There are two parametric models - P-Resp and P-IDR.

Parametric models are written in R and inferred with Stan.

The Stan code is specified in /trcmed_submission/src/models/parametric/CHOSEN_MODEL/SimpleModelHier.stan

To compile and run the .stan file the script from /trcmed_submission/src/run/parametric/CHOSEN_MODEL/SimpleModelHier.R is used.

To work with them it is better to use HPC cluster and run
```shell
/trcmed_submission/src/run/parametric/run_parametric_triton.sh
```
Inside the shell script the desired parametric model address can be chosen, as well as the output directories.

After the sampling has been done and the results stored in the respective folder, the python code to plot them can be run:
```
python /trcmed_submission/src/plot/parametric/CHOSEN_MODEL.py
```
Results (plots and metrics) are stored in /trcmed_submission/data/results_data/parametric/CHOSEN_MODEL/ folder.


#### Nonparametric models
There are three nonparametric models - GP-Resp, GP-LFM and GP-Conv.

Run nonparametric model:
```
python /trcmed_submission/src/run/non_parametric/CHOSEN_MODEL/run_hierarchical.py
```
Results (plots and metrics) are stored in /trcmed_submission/data/results_data/non_parametric/CHOSEN_MODEL/ folder.
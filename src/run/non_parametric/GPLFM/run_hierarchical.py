import os
import numpy as np
import pandas as pd
import argparse
import warnings
from src.utils.GPLFM.data_preparation import arrays_preparation, times_correction, create_meal_prediction
from src.utils.GPLFM.predict import predict, predict_meal
from src.models.non_parametric.GPLFM.model_hierarchical import HierarchicalModel
from src.models.non_parametric.GPLFM.kernels import get_baseline_kernel, get_treatment_time_meal1_kernel, get_treatment_time_meal2_kernel, get_treatment_time_meal1_kernel_lfm, get_treatment_time_meal2_kernel_lfm
import gpflow as gpf
import tensorflow as tf
from matplotlib import  pyplot as plt

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser('Running hierarchical GP model.')
parser.add_argument('--processed_data', type=str, default='./data/processed_data/',
                    help="Path to save processed data.")
parser.add_argument('--results_data', type=str, default='./data/results_data/non_parametric/GPLFM/',
                    help="Path to save results data.")
parser.add_argument('--treatment_effect_time', type=int, default=3,
                    help="Time of the effect of each treatment.")
parser.add_argument('--period', type=str, default='operation',
                    help="Period, with which we work.")
parser.add_argument('--results_parametric_data', type=str, default='./data/results_data/parametric/',
                    help="Path to save results of parametric modelling.")
parser.add_argument('--results_data_meal', type=str, default='./data/results_data/non_parametric/GPLFM/single_meal/',
                    help="Path to save results data.")

def modelling(df_train, df_test, args):
    """General function for creation of model, making predictions.

    Parameters:
    df_train (pd.DataFrame): Dataframe for training
    df_test (pd.DataFrame): Dataframe for testing
    args (dict): contains input arguments
    """
    x, y, meals, patients, P = arrays_preparation(df_train)
    ids = [id.partition('_')[0] for id in patients]
    x_test, y_test, meals_test, _, _ = arrays_preparation(df_test)

    # Construct model
    model = HierarchicalModel(data=(x, y, meals), T=args.treatment_effect_time,
                              baseline_kernels=[get_baseline_kernel() for _ in range(P)],
                              treatment_base_kernels=[get_treatment_time_meal1_kernel_lfm(), get_treatment_time_meal2_kernel_lfm()],
                              mean_functions=[gpf.mean_functions.Zero()
                                              for _ in range(P)],
                              noise_variance=1.0,
                              separating_interval=200,
                              train_noise=True)

    # Train model
    model = train(model)

    # Predict for training data and receive metrics
    metrics_train = {'RMSE': [], 'M1': [], 'M2': [], 'M5': [], "MAE" : [], "R2":[]}
    metrics_train = predict(model, args, ids, metrics_train, data=(x,y,meals), time='train')

    # Predict for testing data and receive metrics
    metrics_test = {'RMSE': [], 'M1': [], 'M2': [], 'M5': [], "MAE" : [], "R2":[]}
    metrics_test = predict(model, args, ids, metrics_test, data=(x_test, y_test, meals_test), time='test')

    # Save the result metrics
    metrics_train = pd.DataFrame.from_dict(metrics_train, orient='index')
    metrics_train['mean'] = metrics_train.mean(axis=1)
    metrics_train['se'] = metrics_train.std(axis=1)/np.sqrt(P)
    metrics_train.to_csv(args.results_data + "/metrics_train.csv")

    metrics_test = pd.DataFrame.from_dict(metrics_test, orient='index')
    metrics_test['mean'] = metrics_test.mean(axis=1)
    metrics_test['se'] = metrics_test.std(axis=1) / np.sqrt(P)
    metrics_test.to_csv(args.results_data + "/metrics_test.csv")

    # Predictions for one meal
    x_test_meal, meals_test_meal = create_meal_prediction(meals, P)
    predict_meal(model, args, ids, metrics_test, data=(x_test_meal, meals_test_meal), time='test')


def train(model):
    """Trains hierarchical GP model.

    Parameters:
    model (gpf.models.GPR): model to train

    Returns:
    model (gpf.models.GPR): trained GPflow model
    """
    # Train model
    gpf.utilities.print_summary(model)
    opt = gpf.optimizers.Scipy()
    min_logs = opt.minimize(model.training_loss,
                            model.trainable_variables,
                            compile=False,
                            options={"disp": True,
                                     "maxiter": 2000})
    gpf.utilities.print_summary(model)

    return model


if __name__ == "__main__":
    args = parser.parse_args()
    os.chdir("../../../../")

    # Downloading processed data
    df_train = pd.read_csv(args.processed_data + '/df_sliced.csv')
    df_test = pd.read_csv(args.processed_data + '/df_sliced_test.csv')

    # Modelling (Train hierarchical GP model and make predictions on glucose data)
    modelling(df_train, df_test, args)



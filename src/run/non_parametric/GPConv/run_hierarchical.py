import pandas as pd
import argparse
import warnings
from src.utils.GPConv.data_preparation import arrays_preparation, times_correction, create_meal_prediction, remove_meal
from src.utils.GPConv.data_simulation import simulate_artificial_data
from src.utils.GPConv.predict import predict, predict_meal, predict_conv
from src.models.non_parametric.GPConv.model_hierarchical import HierarchicalModel
from src.models.non_parametric.GPConv.kernels import get_baseline_kernel
import gpflow as gpf
import numpy as np
import os

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser('Running hierarchical GP model.')
parser.add_argument('--processed_data', type=str, default='./data/processed_data/',
                    help="Path to save processed data.")
parser.add_argument('--results_data', type=str, default='./data/results_data/non_parametric/GPConv/',
                    help="Path to save results data.")
parser.add_argument('--treatment_effect_time', type=int, default=3.0,
                    help="Time of the effect of each treatment.")
parser.add_argument('--period', type=str, default='operation',
                    help="Period, with which we work.")
parser.add_argument('--results_parametric_data', type=str, default='./data/results_data/parametric/',
                    help="Path to save results of parametric modelling.")
parser.add_argument('--meal_type', type=str, default='carbs',
                    help="Type of meal.")
parser.add_argument('--results_data_meal', type=str, default='./data/results_data/non_parametric/GPConv/single_meal/',
                    help="Path to save results data.")

def modelling(df_train, df_test, args):
    """General function for creation of model, making predictions.

    Parameters:
    df_train (pd.DataFrame): Dataframe for training
    df_test (pd.DataFrame): Dataframe for testing
    args (dict): contains input arguments
    """
    x, y, meals, patients, P = arrays_preparation(df_train)
    meals_wo_fat = remove_meal(meals)
    ids = [id.partition('_')[0] for id in patients]
    x_test, y_test, meals_test, _, _ = arrays_preparation(df_test)
    meals_wo_fat_test = remove_meal(meals_test)

    # Construct model with fat
    model_full = HierarchicalModel(data=(x, y, meals), T=args.treatment_effect_time,
                              baseline_kernels=[get_baseline_kernel() for _ in range(P)],
                              mean_functions=[gpf.mean_functions.Zero()
                                              for _ in range(P)],
                              noise_variance=1.0,
                              separating_interval=200,
                              train_noise=True)

    # Construct model without fat
    # model_wo_meal = HierarchicalModel(data=(x, y, meals_wo_fat), T=args.treatment_effect_time,
    #                           baseline_kernels=[get_baseline_kernel() for _ in range(P)],
    #                           mean_functions=[gpf.mean_functions.Zero()
    #                                           for _ in range(P)],
    #                           noise_variance=1.0,
    #                           separating_interval=200,
    #                           train_noise=True)

    # Train models
    model_full = train(model_full)
    #model_wo_meal = train(model_wo_meal)

    # # Predict for training data and receive metrics
    metrics_train = {'RMSE': [], 'M1': [], 'M2': [], 'M5': [], "MAE" : [], "R2":[]}
    metrics_train = predict(model_full, args, ids, metrics_train, data=(x,y,meals), time='train')

    # # Predict for testing data and receive metrics
    metrics_test = {'RMSE': [], 'M1': [], 'M2': [], 'M5': [], "MAE" : [], "R2":[]}
    metrics_test = predict(model_full, args, ids, metrics_test, data=(x_test, y_test, meals_test), time='test')

    # Save the result metrics
    metrics_train = pd.DataFrame.from_dict(metrics_train, orient='index')
    metrics_train['mean'] = metrics_train.mean(axis=1)
    metrics_train['se'] = metrics_train.std(axis=1) / np.sqrt(P)
    os.makedirs(args.results_data + '/' + args.meal_type + '/', exist_ok=True)
    metrics_train.to_csv(args.results_data + '/' + args.meal_type + "/metrics_train.csv")

    metrics_test = pd.DataFrame.from_dict(metrics_test, orient='index')
    metrics_test['mean'] = metrics_test.mean(axis=1)
    metrics_test['se'] = metrics_test.std(axis=1) / np.sqrt(P)
    metrics_test.to_csv(args.results_data + '/' + args.meal_type + "/metrics_test.csv")

    # Predictions for one model on two different datasets
    predict_conv(model_full, args, ids, data=(x_test, y_test, meals_wo_fat_test, meals_test), time='test', full=True)
    x_test_meal, meals_test_meal = create_meal_prediction(meals, P)
    meals_wo_fat_test_meal = remove_meal(meals_test_meal)
    predict_conv(model_full, args, ids, data=(x_test_meal, None, meals_wo_fat_test_meal, meals_test_meal), time='test', full=False)


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



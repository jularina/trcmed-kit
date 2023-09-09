import os
import numpy as np
import pandas as pd
import argparse
import warnings
from src.utils.GPLFM.data_preparation import arrays_preparation, times_correction, create_meal_prediction, make_folds, combine_folds
from src.utils.GPLFM.predict import predict, predict_meal
from src.models.non_parametric.GPLFM.model_hierarchical import HierarchicalModel
from src.models.non_parametric.GPLFM.kernels import get_baseline_kernel, get_treatment_time_meal1_kernel, get_treatment_time_meal2_kernel, get_treatment_time_meal1_kernel_lfm, get_treatment_time_meal2_kernel_lfm
import gpflow as gpf
import itertools
import tensorflow as tf
from matplotlib import  pyplot as plt

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser('Running hierarchical GP model.')
parser.add_argument('--processed_data', type=str, default='./data/real/processed_data/',
                    help="Path to save processed data.")
parser.add_argument('--results_data', type=str, default='./data/real/results_data/non_parametric/GPLFM/',
                    help="Path to save results data.")
parser.add_argument('--treatment_effect_time', type=int, default=3,
                    help="Time of the effect of each treatment.")
parser.add_argument('--period', type=str, default='operation',
                    help="Period, with which we work.")
parser.add_argument('--data', type=str, default='real',
                    help="Data type.")
parser.add_argument('--noise_var', type=float, default=1.0,
                    help="Noise variance for the model.")
parser.add_argument('--results_parametric_data', type=str, default='./data/real/results_data/parametric/PIDR/',
                    help="Path to save results of parametric modelling.")
parser.add_argument('--results_data_meal', type=str, default='./data/real/results_data/non_parametric/GPLFM/single_meal/',
                    help="Path to save results data.")
parser.add_argument('--original_arrays_path', type=str, default='./data/real/processed_data/patients_arrays/',
                    help="Path to numpy arrays with patients data.")
parser.add_argument('--created_arrays_path', type=str, default='./data/real/results_data/non_parametric/GPLFM/patients_arrays/',
                    help="Path to numpy arrays with patients data.")
parser.add_argument('--cross_val', type=bool, default=False,
                    help="Usage of cross-validation.")

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

    # If we want to make cross-validation (4 folds with 3 patients 2-day data in each fold)
    if args.cross_val:
        treatment1_base_kernels_l = [0.25,0.3,0.35]
        treatment2_base_kernels_l = [0.7,0.8,0.85]
        params = [treatment1_base_kernels_l, treatment2_base_kernels_l]
        for element in itertools.product(*params):
            metrics_test = {'RMSE': [], 'M2': [], "MAE": [], "NLL": []}
            x_folded, y_folded, meals_folded = make_folds(x,y,meals,P)

            xs, ys, mealses, xs_val, ys_val, mealses_val = [],[],[],[],[],[]
            # 1st CV step
            x_train_1, y_train_1, meals_train_1 = [x_folded[p][0] for p in range(P)], [y_folded[p][0] for p in range(P)], [
                meals_folded[p][0] for p in range(P)]
            x_val_1, y_val_1, meals_val_1 = [x_folded[p][1] for p in range(P)], [y_folded[p][1] for p in range(P)], [
                meals_folded[p][1] for p in range(P)]
            xs.append(x_train_1)
            ys.append(y_train_1)
            mealses.append(meals_train_1)
            xs_val.append(x_val_1)
            ys_val.append(y_val_1)
            mealses_val.append(meals_val_1)

            # 2nd CV step
            idx = [2,3]
            x_train_2, y_train_2, meals_train_2 = combine_folds(x_folded, y_folded, meals_folded, idx, P)
            x_val_2, y_val_2, meals_val_2 = [x_folded[p][2] for p in range(P)], [y_folded[p][2] for p in range(P)], [
                meals_folded[p][2] for p in range(P)]
            xs.append(x_train_2)
            ys.append(y_train_2)
            mealses.append(meals_train_2)
            xs_val.append(x_val_2)
            ys_val.append(y_val_2)
            mealses_val.append(meals_val_2)

            # 3d CV step
            idx = [3]
            x_train_3, y_train_3, meals_train_3 = combine_folds(x_folded, y_folded, meals_folded, idx, P)
            x_val_3, y_val_3, meals_val_3 = [x_folded[p][3] for p in range(P)], [y_folded[p][3] for p in range(P)], [
                meals_folded[p][3] for p in range(P)]
            xs.append(x_train_3)
            ys.append(y_train_3)
            mealses.append(meals_train_3)
            xs_val.append(x_val_3)
            ys_val.append(y_val_3)
            mealses_val.append(meals_val_3)

            for i in range(3):
                metrics_test_fold = {'RMSE': [], 'M2': [], "MAE": [], "NLL": []}
                x_train, y_train, meals_train = xs[i], ys[i], mealses[i]
                x_val, y_val, meals_val = xs_val[i], ys_val[i], mealses_val[i]

                # Construct model
                model = HierarchicalModel(data=(x, y, meals), T=args.treatment_effect_time,
                                          baseline_kernels=[get_baseline_kernel() for _ in range(P)],
                                          treatment_base_kernels=[get_treatment_time_meal1_kernel_lfm(l=element[0]),
                                                                  get_treatment_time_meal2_kernel_lfm(l=element[1])],
                                          mean_functions=[gpf.mean_functions.Zero()
                                                          for _ in range(P)],
                                          noise_variance=args.noise_var,
                                          separating_interval=200,
                                          train_noise=True)

                # Train model
                model = train(model)

                # Predict for testing data and receive metrics
                metrics_test_fold = predict(model, args, ids, metrics_test_fold, data=(x_val, y_val, meals_val), time='test')

                metrics_test_fold = pd.DataFrame.from_dict(metrics_test_fold, orient='index')
                metrics_test_fold['mean'] = metrics_test_fold.mean(axis=1)

                metrics_test['RMSE'].append(metrics_test_fold.loc['RMSE','mean'])
                metrics_test['M2'].append(metrics_test_fold.loc['M2','mean'])
                metrics_test['MAE'].append(metrics_test_fold.loc['MAE','mean'])
                metrics_test['NLL'].append(metrics_test_fold.loc['NLL','mean'])

            metrics_test = pd.DataFrame.from_dict(metrics_test, orient='index')
            metrics_test['mean'] = metrics_test.mean(axis=1)
            metrics_test['sd'] = metrics_test.std(axis=1)
            metrics_test['se'] = metrics_test.std(axis=1) / np.sqrt(3)
            metrics_test.to_csv(args.results_data + "/metrics_test_cv_"+str(element[0])+"_"+str(element[1])+"_"+str(element[2])+"_"+str(element[3])+".csv")
            print(element)
            print(metrics_test)

    else:

        # Construct model
        model = HierarchicalModel(data=(x, y, meals), T=args.treatment_effect_time,
                                  baseline_kernels=[get_baseline_kernel() for _ in range(P)],
                                  treatment_base_kernels=[get_treatment_time_meal1_kernel_lfm(l=0.3), get_treatment_time_meal2_kernel_lfm(l=0.8)],
                                  mean_functions=[gpf.mean_functions.Zero()
                                                  for _ in range(P)],
                                  noise_variance=args.noise_var,
                                  separating_interval=200,
                                  train_noise=True)

        # Train model
        model = train(model)

        # Predict for training data and receive metrics
        metrics_train = {'RMSE': [], 'M2': [], "MAE" : [], "NLL":[]}
        metrics_train = predict(model, args, ids, metrics_train, data=(x,y,meals), time='train')

        # Predict for testing data and receive metrics
        metrics_test = {'RMSE': [], 'M2': [], "MAE" : [], "NLL":[]}
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

    if args.data == 'real':
        # For real data make time corrections, based on learnt parametric models
        time_corr_df = pd.read_csv(
            args.results_parametric_data + 'time_corrections_train_test.csv',
            index_col=None)
        df_train, df_test = times_correction(df_train, time_corr_df, df_test, args)

        # Select ids
        ids = ['31_2', '12_2', '32_2', '46_2', '29_2', '57_2', '23_2', '9_2', '28_2', '76_1', '65_1', '60_1']
        # ids = ['46_2','12_2','23_2','29_2','28_2']
        # ids = ['12_2', '29_2', '28_2']
        df_train = df_train[df_train['id'].isin(ids)]
        df_test = df_test[df_test['id'].isin(ids)]

    # Modelling (Train hierarchical GP model and make predictions on glucose data)
    modelling(df_train, df_test, args)



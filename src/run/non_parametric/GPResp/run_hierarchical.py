import os
import numpy as np
import pandas as pd
import argparse
import warnings
from src.utils.GPResp.data_preparation import arrays_preparation, times_correction, create_meal_prediction, \
    patients_data_arrays, patients_data_arrays_onemeal, make_folds
from src.utils.GPResp.predict import predict, predict_meal, predict_meal_severalsetups
from src.models.non_parametric.GPResp.model_hierarchical import HierarchicalModel
from src.models.non_parametric.GPResp.kernels import get_baseline_kernel, get_treatment_time_meal1_kernel, \
    get_treatment_time_meal2_kernel, get_treatment_time_meal1_kernel_lfm, get_treatment_time_meal2_kernel_lfm
import gpflow as gpf
import itertools

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser('Running hierarchical GP model.')
parser.add_argument('--processed_data', type=str, default='./data/real/processed_data/',
                    help="Path to save processed data.")
parser.add_argument('--results_data', type=str, default='./data/real/results_data/non_parametric/GPResp/',
                    help="Path to save results data.")
parser.add_argument('--results_data_meal', type=str,
                    default='./data/real/results_data/non_parametric/GPResp/single_meal/',
                    help="Path to save results data.")
parser.add_argument('--treatment_effect_time', type=int, default=3.0,
                    help="Time of the effect of each treatment.")
parser.add_argument('--period', type=str, default='operation',
                    help="Period, with which we work.")
parser.add_argument('--data', type=str, default='real',
                    help="Data type.")
parser.add_argument('--noise_var', type=float, default=1.0,
                    help="Noise variance for the model.")
parser.add_argument('--results_parametric_data', type=str, default='./data/real/results_data/parametric/PIDR/',
                    help="Path to save results of parametric modelling.")
parser.add_argument('--original_arrays_path', type=str, default='./data/real/processed_data/patients_arrays/',
                    help="Path to numpy arrays with patients data.")
parser.add_argument('--created_arrays_path', type=str, default='./data/real/results_data/non_parametric/GPResp/patients_arrays/',
                    help="Path to numpy arrays with patients data.")
parser.add_argument('--cross_val', type=bool, default=True,
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
        treatment1_base_kernels_v, treatment1_base_kernels_l = [0.8,1.0,1.2], [0.25,0.3,0.35]
        treatment2_base_kernels_v, treatment2_base_kernels_l = [0.07,0.1,0.15], [0.7,0.8,0.85]
        params = [treatment1_base_kernels_v, treatment1_base_kernels_l, treatment2_base_kernels_v, treatment2_base_kernels_l]

        for element in itertools.product(*params):
            metrics_test = {'RMSE': [], 'M2': [], "MAE": [], "NLL": []}
            x_folded, y_folded, meals_folded = make_folds(x,y,meals,P)

            for i in range(4):
                x_train, y_train = [x_folded[p][:i]+x_folded[p][i+1:] for p in range(P)], [y_folded[p][:i]+y_folded[p][i+1:] for p in range(P)]
                meals_train = [np.concatenate(meals_folded[p].pop(i), axis=0) for p in range(P)]
                x_val, y_val, meals_val = [x_folded[p][i] for p in range(P)], [y_folded[p][i] for p in range(P)], [meals_folded[p][i] for p in range(P)]

                # Construct model
                model = HierarchicalModel(data=(x_train, y_train, meals_train), T=args.treatment_effect_time,
                                          baseline_kernels=[get_baseline_kernel() for _ in range(P)],
                                          treatment_base_kernels=[get_treatment_time_meal1_kernel(v=element[0],l=element[1]),
                                                                  get_treatment_time_meal2_kernel(v=element[2],l=element[3])],
                                          mean_functions=[gpf.mean_functions.Zero()
                                                          for _ in range(P)],
                                          noise_variance=args.noise_var,
                                          separating_interval=200,
                                          train_noise=True)

                # Train model
                model = train(model)

                # Predict for testing data and receive metrics
                metrics_test = predict(model, args, ids, metrics_test, data=(x_val, y_val, meals_val), time='test')

            metrics_test = pd.DataFrame.from_dict(metrics_test, orient='index')
            metrics_test['mean'] = metrics_test.mean(axis=1)
            metrics_test['se'] = metrics_test.std(axis=1) / np.sqrt(metrics_test.columns-1)
            print(element)
            print(metrics_test)


    else:
        if args.data == 'real':
            patients_data_arrays(x, y, meals, x_test, y_test, meals_test, ids, args.original_arrays_path)

        # Construct model
        model = HierarchicalModel(data=(x, y, meals), T=args.treatment_effect_time,
                                  baseline_kernels=[get_baseline_kernel() for _ in range(P)],
                                  treatment_base_kernels=[get_treatment_time_meal1_kernel(v=1.0,l=0.3),
                                                          get_treatment_time_meal2_kernel(v=0.1,l=0.8)],
                                  mean_functions=[gpf.mean_functions.Zero()
                                                  for _ in range(P)],
                                  noise_variance=args.noise_var,
                                  separating_interval=200,
                                  train_noise=True)

        # Train model
        model = train(model)

        # Predict for training data and receive metrics
        metrics_train = {'RMSE': [], 'M2': [], "MAE": [], "NLL": []}
        metrics_train = predict(model, args, ids, metrics_train, data=(x, y, meals), time='train')

        # Predict for testing data and receive metrics
        metrics_test = {'RMSE': [], 'M2': [], "MAE": [], "NLL": []}
        metrics_test = predict(model, args, ids, metrics_test, data=(x_test, y_test, meals_test), time='test')

        # Save the result metrics
        metrics_train = pd.DataFrame.from_dict(metrics_train, orient='index')
        metrics_train['mean'] = metrics_train.mean(axis=1)
        metrics_train['se'] = metrics_train.std(axis=1) / np.sqrt(P)
        metrics_train.to_csv(args.results_data + "/metrics_train.csv")

        metrics_test = pd.DataFrame.from_dict(metrics_test, orient='index')
        metrics_test['mean'] = metrics_test.mean(axis=1)
        metrics_test['se'] = metrics_test.std(axis=1) / np.sqrt(P)
        metrics_test.to_csv(args.results_data + "/metrics_test.csv")

        # Predictions for one meal
        x_test_meal, meals_test_meal, meals_test_meal_same, meals_test_meal_reverse = create_meal_prediction(meals, P)
        predict_meal(model, args, ids, metrics_test, data=(x_test_meal, meals_test_meal), time='test', order='original')
        predict_meal(model, args, ids, metrics_test, data=(x_test_meal, meals_test_meal_same), time='test', order='same')
        predict_meal(model, args, ids, metrics_test, data=(x_test_meal, meals_test_meal_reverse), time='test',
                     order='reverse')
        predict_meal_severalsetups(model, args, ids,
                                   data=(x_test_meal, meals_test_meal, meals_test_meal_same, meals_test_meal_reverse))

        if args.data == 'real':
            patients_data_arrays_onemeal(x_test_meal, meals_test_meal, ids, args.original_arrays_path)



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
        #ids = ['12_2', '29_2', '28_2']
        df_train = df_train[df_train['id'].isin(ids)]
        df_test = df_test[df_test['id'].isin(ids)]

    # Modelling (Train hierarchical GP model and make predictions on glucose data)
    modelling(df_train, df_test, args)

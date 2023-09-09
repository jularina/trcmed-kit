import pandas as pd
import argparse
import warnings
from src.utils.GPConv.data_preparation import arrays_preparation, times_correction, create_meal_prediction, remove_meal, \
    make_folds, combine_folds
from src.utils.GPConv.data_simulation import simulate_artificial_data
from src.utils.GPConv.predict import predict, predict_meal, predict_conv
from src.models.non_parametric.GPConv.model_hierarchical import HierarchicalModel
from src.models.non_parametric.GPConv.kernels import get_baseline_kernel
import gpflow as gpf
import numpy as np
import os

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser('Running hierarchical GP model.')
parser.add_argument('--processed_data', type=str, default='./data/real/processed_data/',
                    help="Path to save processed data.")
parser.add_argument('--results_data', type=str, default='./data/real/results_data/non_parametric/GPConv/',
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
parser.add_argument('--meal_type', type=str, default='carbs',
                    help="Type of meal.")
parser.add_argument('--results_data_meal', type=str,
                    default='./data/real/results_data/non_parametric/GPConv/single_meal/',
                    help="Path to save results data.")
parser.add_argument('--original_arrays_path', type=str, default='./data/real/processed_data/patients_arrays/',
                    help="Path to numpy arrays with patients data.")
parser.add_argument('--created_arrays_path', type=str,
                    default='./data/real/results_data/non_parametric/GPConv/patients_arrays/',
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
    meals_wo_fat = remove_meal(meals)
    ids = [id.partition('_')[0] for id in patients]
    x_test, y_test, meals_test, _, _ = arrays_preparation(df_test)
    meals_wo_fat_test = remove_meal(meals_test)

    # If we want to make cross-validation (4 folds with 3 patients 2-day data in each fold)
    if args.cross_val:
        treatment1_base_kernels_l = [0.2, 0.3, 0.4, 0.45]
        for element in treatment1_base_kernels_l:
            metrics_test = {'RMSE': [], 'M2': [], "MAE": [], "NLL": []}
            x_folded, y_folded, meals_folded = make_folds(x, y, meals, P)

            xs, ys, mealses, xs_val, ys_val, mealses_val = [], [], [], [], [], []
            # 1st CV step
            x_train_1, y_train_1, meals_train_1 = [x_folded[p][0] for p in range(P)], [y_folded[p][0] for p in
                                                                                       range(P)], [
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
            idx = [2, 3]
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
                model_full = HierarchicalModel(data=(x_train, y_train, meals_train), T=args.treatment_effect_time,
                                               baseline_kernels=[get_baseline_kernel() for _ in range(P)],
                                               mean_functions=[gpf.mean_functions.Zero()
                                                               for _ in range(P)],
                                               l=element,
                                               noise_variance=args.noise_var,
                                               separating_interval=200,
                                               train_noise=True)

                # Train model
                model = train(model_full)

                # Predict for testing data and receive metrics
                metrics_test_fold = predict(model, args, ids, metrics_test_fold, data=(x_val, y_val, meals_val),
                                            time='test')

                metrics_test_fold = pd.DataFrame.from_dict(metrics_test_fold, orient='index')
                metrics_test_fold['mean'] = metrics_test_fold.mean(axis=1)

                metrics_test['RMSE'].append(metrics_test_fold.loc['RMSE', 'mean'])
                metrics_test['M2'].append(metrics_test_fold.loc['M2', 'mean'])
                metrics_test['MAE'].append(metrics_test_fold.loc['MAE', 'mean'])
                metrics_test['NLL'].append(metrics_test_fold.loc['NLL', 'mean'])

            metrics_test = pd.DataFrame.from_dict(metrics_test, orient='index')
            metrics_test['mean'] = metrics_test.mean(axis=1)
            metrics_test['sd'] = metrics_test.std(axis=1)
            metrics_test['se'] = metrics_test.std(axis=1) / np.sqrt(3)
            metrics_test.to_csv(
                args.results_data + "/carbs/metrics_test_cv_" + str(element)+ ".csv")
            print(element)
            print(metrics_test)


    else:

        # Construct model with fat
        model_full = HierarchicalModel(data=(x, y, meals), T=args.treatment_effect_time,
                                       baseline_kernels=[get_baseline_kernel() for _ in range(P)],
                                       mean_functions=[gpf.mean_functions.Zero()
                                                       for _ in range(P)],
                                       l=0.4,
                                       noise_variance=args.noise_var,
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
        # model_wo_meal = train(model_wo_meal)

        # # Predict for training data and receive metrics
        metrics_train = {'RMSE': [], 'M2': [], "MAE": [], "NLL": []}
        metrics_train = predict(model_full, args, ids, metrics_train, data=(x, y, meals), time='train')

        # # Predict for testing data and receive metrics
        metrics_test = {'RMSE': [], 'M2': [], "MAE": [], "NLL": []}
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
        predict_conv(model_full, args, ids, data=(x_test, y_test, meals_wo_fat_test, meals_test), time='test',
                     full=True)
        x_test_meal, meals_test_meal = create_meal_prediction(meals, P)
        meals_wo_fat_test_meal = remove_meal(meals_test_meal)
        predict_conv(model_full, args, ids, data=(x_test_meal, None, meals_wo_fat_test_meal, meals_test_meal),
                     time='test', full=False)


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

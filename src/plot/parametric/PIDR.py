import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score
from scipy import stats
import math
import os
import argparse
import warnings

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser('Plotting results from stan model.')
parser.add_argument('--tasks', type=int, default=1, help="Number of times the model has been run in cluster.")
parser.add_argument('--results_data', type=str, default='./data/results_data/parametric/PIDR/',
                    help="Folder for stan results storage.")
parser.add_argument('--processed_data', type=str, default='./data/processed_data/',
                    help="Path to save processed data.")
parser.add_argument('--period', type=str, default='operation',
                    help="Time period to be considered.")
parser.add_argument('--model', type=int, default=3, help="Index of running model.")
parser.add_argument('--modelname', type=str, default='P-IDR', help="Index of running model.")
parser.add_argument('--modelstr', type=str, default='model3',
                    help="Model name.")


def analyse_train_results(path, P, patients, df_sliced, trend_p, N, tasks=1):
    """Analyse and plot results from fitting training data to stan model.

    Parameters:
    path (str): Path to results data folder
    P (int): number of patients
    patients (np.ndarray): Numpy array with patients indices
    trend_p (np.ndarray): Numpy array with trend values for each patient
    df_sliced (pd.Dataframe): cut train dataframe of patients' meals/glucose data
    N (np.ndarray): Numpy array with numbers of glucose observations for each patient
    tasks (int): Number of tasks

    Returns:
    metrics_train (dict): Dictionary to store training metrics
    rmse_conf_int (np.ndarray): Numpy array to store confidence intervals for RMSE
   """

    metrics_train = {'RMSE': [], 'M2': [], 'M5': [], "MAE" : [], "R2":[]}
    rmse_conf_int = np.zeros((P * tasks, 5))

    for t in range(tasks):
        samples_y = pd.read_csv(path + 'samples_y_' + str(t) + '.csv')
        samples_y1 = pd.read_csv(path + 'samples_y1_' + str(t) + '.csv')
        samples_y2 = pd.read_csv(path + 'samples_y2_' + str(t) + '.csv')

        for idx in range(P):
            patient_id = patients[idx]
            df = df_sliced[df_sliced['id'] == patient_id]
            df_ys = df[~df['y'].isna()]

            # Predictions retrieval
            trend = [trend_p[idx]] * N[idx]
            meals1 = samples_y1.iloc[idx, :N[idx]]
            meals2 = samples_y2.iloc[idx, :N[idx]]
            overall_glucose = samples_y.iloc[idx, :N[idx]]

            # Metrics calculation
            rmse = np.sqrt(mean_squared_error(overall_glucose, df_ys['y']))
            mae_score = mae(overall_glucose, df_ys['y'])
            r2 = r2_score(overall_glucose, df_ys['y'])
            M1 = np.var(trend) / np.var(df_ys['y'])
            M2 = np.var(overall_glucose) / np.var(df_ys['y']) - M1
            M5 = abs(np.var(np.array(meals1) + np.array(meals2)) - np.var(df_ys['y']))

            # Appending metrics to the metrics dictionary
            metrics_train['RMSE'].append(rmse)
            metrics_train['M2'].append(M2)
            metrics_train['M5'].append(M5)
            metrics_train['MAE'].append(mae_score)
            metrics_train['R2'].append(r2)

            # Confidence interval
            n = len(df_ys['y'])
            c1, c2 = stats.chi2.ppf([0.025, 1 - 0.025], n)
            upper, lower = math.sqrt(n / c1) * rmse, math.sqrt(n / c2) * rmse
            rmse_conf_int[P * t + idx, :] = [idx, round(lower, 3), round(upper, 3), round(rmse, 3),
                                             round(upper, 3) - round(lower, 3)]

            # Plotting
            fig, axs = plt.subplots(2, 1, figsize=(20, 10))
            fig.suptitle(
                "Glucose level for patient {id} with training RMSE = {trerr} (lower={lower},upper={upper}), M2 = {M2}, M5 = {M5}.".format(
                    id=patient_id, trerr=round(rmse, 3), lower=round(lower, 3), upper=round(upper, 3),
                    M2=round(M2, 3), M5=round(M5, 3)))

            axs[0].plot(df_ys['t'], df_ys['y'], 'kx', ms=5, alpha=0.5, label='True observations')
            axs[0].plot(df_ys['t'], meals1, c="mediumpurple", linewidth=2, label='Posterior mean for 1st meal',
                        zorder=2)
            axs[0].plot(df_ys['t'], meals2, c="cornflowerblue", linewidth=2, label='Posterior mean for 2nd meal',
                        zorder=2)
            axs[0].plot(df_ys['t'], trend, c="purple", linewidth=2, label='Baseline', zorder=2)
            axs[0].plot(df_ys['t'], overall_glucose, c="midnightblue", linewidth=2, label='Fitted glucose curve',
                        zorder=2)
            axs[0].set(ylabel="Glucose")
            axs[0].legend(loc='upper right')

            axs[1].bar(df['t'], df['CARBS'], color='mediumpurple', width=0.3, label="Carbs")
            axs[1].bar(df['t'], df['FAT'], bottom=df['CARBS'], color='cornflowerblue', width=0.3,
                       label='Fat')
            axs[1].set(xlabel="Time (hours)", ylabel="Stacked meals")
            axs[1].legend()

            patient_id = patient_id.partition('_')[0]
            if not os.path.exists(path + 'id' + patient_id + '/'):
                os.makedirs(path + 'id' + patient_id + '/')
            plt.savefig(path + 'id' + patient_id + '/predictions_train_task' + str(t) + '.pdf')

    metrics_train = pd.DataFrame.from_dict(metrics_train, orient='index')
    metrics_train['mean'] = metrics_train.mean(axis=1)
    metrics_train['se'] = metrics_train.std(axis=1) / np.sqrt(P)
    metrics_train.to_csv(path + "metrics_train.csv")
    rmse_conf_df = pd.DataFrame(rmse_conf_int)
    rmse_conf_df.groupby(rmse_conf_df.columns[0]).mean().to_csv(path + "confintervals_train.csv")

    return metrics_train, rmse_conf_int


def model1_test_results(fitted_params_patients, N_max_test, M_max_test, tx_test, t_test, x1_test, x2_test):
    """Combining model1 results from fitting testing data to stan model.

    Parameters:
    fitted_params_patients (pd.Dataframe): Dataframe with fitted stan parameters for each patiemt
    N_max_test (int): maximum number of glucose observations
    M_max_test (int): maximum number of meals
    tx_test (np.ndarray): Numpy array with true meals times for each patient
    t_test (np.ndarray): Numpy array with observed meals times for each patient
    x1_test (np.ndarray): Numpy array with 1st meal values for each patient at each time
    x2_test (np.ndarray): Numpy array with 2nd meal values for each patient at each time

    Returns:
    samples_y_test (pd.Dataframe): Dataframe with sampling results for combined response
    samples_y1_test (pd.Dataframe): Dataframe with sampling results for 1st meal
    samples_y2_test (pd.Dataframe): Dataframe with sampling results for 2nd meal
   """
    alpha_p = fitted_params_patients.iloc[:, 0]
    beta1_p = fitted_params_patients.iloc[:, 1]
    beta2_p = fitted_params_patients.iloc[:, 2]
    Mcumsum_test = np.cumsum(M_test)
    Mcumsum_test = np.insert(Mcumsum_test, 0, 0)

    resp1, resp2, resp = np.zeros((P, N_max_test, M_max_test)), np.zeros((P, N_max_test, M_max_test)), np.zeros(
        (P, N_max_test, M_max_test))
    samples_y_test = np.zeros((P, N_max_test))
    samples_y1_test = np.zeros((P, N_max_test))
    samples_y2_test = np.zeros((P, N_max_test))

    for p in range(P):
        for n in range(N_test[p]):
            for m in range(M_test[p]):
                if abs((tx_test[Mcumsum_test[p] + m] - t_test[p, n]) < 250):
                    resp1[p, n, m] = (x1_test[Mcumsum_test[p] + m] * beta1_p[p]) * np.exp(
                        -0.5 * (t_test[p, n] - tx_test[Mcumsum_test[p] + m] - 3 * (alpha_p[p])) ** 2 / (
                            alpha_p[p]) ** 2)
                    resp2[p, n, m] = (x2_test[Mcumsum_test[p] + m] * beta2_p[p]) * np.exp(
                        -0.5 * (t_test[p, n] - tx_test[Mcumsum_test[p] + m] - 3 * (alpha_p[p])) ** 2 / (
                            alpha_p[p]) ** 2)
                    resp[p, n, m] = resp1[p, n, m] + resp2[p, n, m]

            samples_y_test[p, n] = sum(resp[p, n,]) + trend_p[p]
            samples_y1_test[p, n] = sum(resp1[p, n,])
            samples_y2_test[p, n] = sum(resp2[p, n,])

    samples_y_test = pd.DataFrame(samples_y_test)
    samples_y1_test = pd.DataFrame(samples_y1_test)
    samples_y2_test = pd.DataFrame(samples_y2_test)

    return samples_y_test, samples_y1_test, samples_y2_test


def model2_test_results(fitted_params_patients, N_max_test, M_max_test, tx_test, t_test, x1_test, x2_test):
    """Combining model2 results from fitting testing data to stan model.

    Parameters:
    fitted_params_patients (pd.Dataframe): Dataframe with fitted stan parameters for each patiemt
    N_max_test (int): maximum number of glucose observations
    M_max_test (int): maximum number of meals
    tx_test (np.ndarray): Numpy array with true meals times for each patient
    t_test (np.ndarray): Numpy array with observed meals times for each patient
    x1_test (np.ndarray): Numpy array with 1st meal values for each patient at each time
    x2_test (np.ndarray): Numpy array with 2nd meal values for each patient at each time

    Returns:
    samples_y_test (pd.Dataframe): Dataframe with sampling results for combined response
    samples_y1_test (pd.Dataframe): Dataframe with sampling results for 1st meal
    samples_y2_test (pd.Dataframe): Dataframe with sampling results for 2nd meal
   """
    alpha1_p = fitted_params_patients.iloc[:, 0]
    alpha2_p = fitted_params_patients.iloc[:, 1]
    beta1_p = fitted_params_patients.iloc[:, 2]
    beta2_p = fitted_params_patients.iloc[:, 3]
    Mcumsum_test = np.cumsum(M_test)
    Mcumsum_test = np.insert(Mcumsum_test, 0, 0)

    resp1, resp2, resp = np.zeros((P, N_max_test, M_max_test)), np.zeros((P, N_max_test, M_max_test)), np.zeros(
        (P, N_max_test, M_max_test))
    samples_y_test = np.zeros((P, N_max_test))
    samples_y1_test = np.zeros((P, N_max_test))
    samples_y2_test = np.zeros((P, N_max_test))

    for p in range(P):
        for n in range(N_test[p]):
            for m in range(M_test[p]):
                if abs((tx_test[Mcumsum_test[p] + m] - t_test[p, n]) < 250):
                    resp1[p, n, m] = (x1_test[Mcumsum_test[p] + m] * beta1_p[p]) * np.exp(
                        -0.5 * (t_test[p, n] - tx_test[Mcumsum_test[p] + m] - 3 * (alpha1_p[p])) ** 2 / (
                            alpha1_p[p]) ** 2)
                    resp2[p, n, m] = (x2_test[Mcumsum_test[p] + m] * beta2_p[p]) * np.exp(
                        -0.5 * (t_test[p, n] - tx_test[Mcumsum_test[p] + m] - 3 * (alpha2_p[p])) ** 2 / (
                            alpha2_p[p]) ** 2)
                    resp[p, n, m] = resp1[p, n, m] + resp2[p, n, m]

            samples_y_test[p, n] = sum(resp[p, n,]) + trend_p[p]
            samples_y1_test[p, n] = sum(resp1[p, n,])
            samples_y2_test[p, n] = sum(resp2[p, n,])

    samples_y_test = pd.DataFrame(samples_y_test)
    samples_y1_test = pd.DataFrame(samples_y1_test)
    samples_y2_test = pd.DataFrame(samples_y2_test)

    return samples_y_test, samples_y1_test, samples_y2_test


def model3_test_results(fitted_params_patients, N_max_test, M_max_test, tx_test, t_test, x1_test, x2_test):
    """Combining model3 results from fitting testing data to stan model.

    Parameters:
    fitted_params_patients (pd.Dataframe): Dataframe with fitted stan parameters for each patiemt
    N_max_test (int): maximum number of glucose observations
    M_max_test (int): maximum number of meals
    tx_test (np.ndarray): Numpy array with true meals times for each patient
    t_test (np.ndarray): Numpy array with observed meals times for each patient
    x1_test (np.ndarray): Numpy array with 1st meal values for each patient at each time
    x2_test (np.ndarray): Numpy array with 2nd meal values for each patient at each time

    Returns:
    samples_y_test (pd.Dataframe): Dataframe with sampling results for combined response
    samples_y1_test (pd.Dataframe): Dataframe with sampling results for 1st meal
    samples_y2_test (pd.Dataframe): Dataframe with sampling results for 2nd meal
   """
    alpha1_p = fitted_params_patients.iloc[:, 0]
    coeff_alpha2_p = fitted_params_patients.iloc[:, 1]
    alpha2_p = alpha1_p * coeff_alpha2_p
    beta1_p = fitted_params_patients.iloc[:, 2]
    beta2_p = fitted_params_patients.iloc[:, 3]
    Mcumsum_test = np.cumsum(M_test)
    Mcumsum_test = np.insert(Mcumsum_test, 0, 0)

    resp1, resp2, resp = np.zeros((P, N_max_test, M_max_test)), np.zeros((P, N_max_test, M_max_test)), np.zeros(
        (P, N_max_test, M_max_test))
    samples_y_test = np.zeros((P, N_max_test))
    samples_y1_test = np.zeros((P, N_max_test))
    samples_y2_test = np.zeros((P, N_max_test))

    for p in range(P):
        for n in range(N_test[p]):
            for m in range(M_test[p]):
                if abs((tx_test[Mcumsum_test[p] + m] - t_test[p, n]) < 250):
                    resp1[p, n, m] = (x1_test[Mcumsum_test[p] + m] * beta1_p[p]) * np.exp(
                        -0.5 * (t_test[p, n] - tx_test[Mcumsum_test[p] + m] - 3 * (alpha1_p[p])) ** 2 / (
                            alpha1_p[p]) ** 2)
                    resp2[p, n, m] = (x2_test[Mcumsum_test[p] + m] * beta2_p[p]) * np.exp(
                        -0.5 * (t_test[p, n] - tx_test[Mcumsum_test[p] + m] - 3 * (alpha2_p[p])) ** 2 / (
                            alpha2_p[p]) ** 2)
                    resp[p, n, m] = resp1[p, n, m] + resp2[p, n, m]

            samples_y_test[p, n] = sum(resp[p, n,]) + trend_p[p]
            samples_y1_test[p, n] = sum(resp1[p, n,])
            samples_y2_test[p, n] = sum(resp2[p, n,])

    samples_y_test = pd.DataFrame(samples_y_test)
    samples_y1_test = pd.DataFrame(samples_y1_test)
    samples_y2_test = pd.DataFrame(samples_y2_test)

    return samples_y_test, samples_y1_test, samples_y2_test


def analyse_test_results(path, P, patients, df_sliced_test, trend_p, N_test, M_test, model, tasks=1):
    """Analyse and plot results from fitting testing data to stan model.

    Parameters:
    path (str): Path to results data folder
    P (int): number of patients
    patients (np.ndarray): Numpy array with patients indices
    trend_p (np.ndarray): Numpy array with trnd values for each patient
    df_sliced_test (pd.Dataframe): cut train dataframe of patients' meals/glucose data
    N_test (np.ndarray): Numpy array with numbers of glucose observations for each patient
    M_test (np.ndarray): Numpy array with numbers of meals for each patient
    model (int): Chosen model
    tasks (int): Number of tasks

    Returns:
    metrics_test (dict): Dictionary to store testing metrics
   """
    metrics_test = {'RMSE': [], 'M2': [], 'M5': [], "MAE" : [], "R2":[]}
    N_max_test, M_max_test = max(N_test), max(M_test)

    for t in range(tasks):
        fitted_params_patients = pd.read_csv(path + 'fitted_params_patients_' + str(t) + '.csv')
        if model == 1:
            samples_y_test, samples_y1_test, samples_y2_test = model1_test_results(fitted_params_patients, N_max_test,
                                                                                   M_max_test, tx_test, t_test, x1_test,
                                                                                   x2_test)
        elif model == 2:
            samples_y_test, samples_y1_test, samples_y2_test = model2_test_results(fitted_params_patients, N_max_test,
                                                                                   M_max_test, tx_test, t_test, x1_test,
                                                                                   x2_test)
        elif model == 3 or model == 4:
            samples_y_test, samples_y1_test, samples_y2_test = model3_test_results(fitted_params_patients, N_max_test,
                                                                                   M_max_test, tx_test, t_test, x1_test,
                                                                                   x2_test)

        for idx in range(P):
            patient_id = patients[idx]
            df = df_sliced_test[df_sliced_test['id'] == patient_id]
            df_ys = df[~df['y'].isna()]

            trend = [trend_p[idx]] * N_test[idx]
            meals1 = samples_y1_test.iloc[idx, :N_test[idx]]
            meals2 = samples_y2_test.iloc[idx, :N_test[idx]]
            overall_glucose = samples_y_test.iloc[idx, :N_test[idx]]

            # Metrics calculation
            if df_ys['y'].size == 0 or overall_glucose.size == 0:
                break
            rmse = np.sqrt(mean_squared_error(overall_glucose, df_ys['y']))
            mae_score = mae(overall_glucose, df_ys['y'])
            r2 = r2_score(overall_glucose, df_ys['y'])
            M1 = np.var(trend) / np.var(df_ys['y'])
            M2 = np.var(overall_glucose) / np.var(df_ys['y']) - M1
            M5 = abs(np.var(np.array(meals1) + np.array(meals2)) - np.var(df_ys['y']))

            # Confidence interval
            n = len(df_ys['y'])
            c1, c2 = stats.chi2.ppf([0.025, 1 - 0.025], n)
            upper, lower = math.sqrt(n / c1) * rmse, math.sqrt(n / c2) * rmse

            # Appending metrics to the metrics dictionary
            metrics_test['RMSE'].append(rmse)
            metrics_test['M2'].append(M2)
            metrics_test['M5'].append(M5)
            metrics_test['MAE'].append(mae_score)
            metrics_test['R2'].append(r2)

            # Plotting
            fig, axs = plt.subplots(2, 1, figsize=(5.2, 4.0), dpi=300, sharex=True)
            mpl.rcParams["figure.autolayout"] = True

            axs[0].plot(df_ys['t'], df_ys['y'], 'kx', ms=5, alpha=0.5, label='True observations')
            axs[0].plot(df_ys['t'], meals1, c="darkmagenta", linewidth=2, label='Posterior mean for carbs', zorder=2)
            axs[0].plot(df_ys['t'], meals2, c="orange", linewidth=2, label='Posterior mean for fat', zorder=2)
            axs[0].plot(df_ys['t'], trend, c="grey", linewidth=2, label='Baseline', zorder=2)
            axs[0].plot(df_ys['t'], overall_glucose, c="royalblue", linewidth=2, label='Fitted glucose curve', zorder=2)
            axs[0].set(ylabel="Glucose (mmol/l)")
            axs[0].set_title('{} glucose response to carbs, fat and baseline trend.'.format(args.modelname),
                             weight='bold')
            axs[0].legend(loc='upper right')

            axs[1].bar(df['t'], df['CARBS'], color='darkmagenta', width=0.3, label="Carbs")
            axs[1].bar(df['t'], df['FAT'], bottom=df['CARBS'], color='orange', width=0.3, label='Fat')
            axs[1].set(xlabel="Time (hours)", ylabel="Stacked meals (g)")
            axs[1].set_title('True meals, eaten after operation.', weight='bold')
            axs[1].set_axisbelow(True)
            axs[1].grid(which='major', color='#DDDDDD', linewidth=0.8)
            axs[1].legend()

            patient_id = patient_id.partition('_')[0]
            plt.subplots_adjust(top=0.93)
            plt.savefig(path + 'id' + patient_id + '/predictions_test_task' + str(t) + '.pdf')

    metrics_test = pd.DataFrame.from_dict(metrics_test, orient='index')
    metrics_test['mean'] = metrics_test.mean(axis=1)
    metrics_test['se'] = metrics_test.std(axis=1) / np.sqrt(P)
    metrics_test.to_csv(path + "metrics_test.csv")

    return metrics_test


def analyse_2models_test_results(path, P, patients, df_sliced_test, trend_p, N_test, M_test, models='models34',
                                 tasks=1):
    """Analyse and plot results from fitting testing data to stan model.

    Parameters:
    path (str): Path to results data folder
    P (int): number of patients
    patients (np.ndarray): Numpy array with patients indices
    trend_p (np.ndarray): Numpy array with trnd values for each patient
    df_sliced_test (pd.Dataframe): cut train dataframe of patients' meals/glucose data
    N_test (np.ndarray): Numpy array with numbers of glucose observations for each patient
    M_test (np.ndarray): Numpy array with numbers of meals for each patient
    model (int): Chosen model
    tasks (int): Number of tasks

    Returns:
    metrics_test (dict): Dictionary to store testing metrics
   """
    N_max_test, M_max_test = max(N_test), max(M_test)

    for t in range(tasks):
        if models == 'models12':
            fitted_params_patients_model1 = pd.read_csv(path + 'model1/fitted_params_patients_' + str(t) + '.csv')
            fitted_params_patients_model2 = pd.read_csv(path + 'model2/fitted_params_patients_' + str(t) + '.csv')
            samples_y_test_model1, samples_y1_test_model1, samples_y2_test_model1 = model1_test_results(
                fitted_params_patients_model1, N_max_test,
                M_max_test, tx_test, t_test, x1_test,
                x2_test)
            samples_y_test_model2, samples_y1_test_model2, samples_y2_test_model2 = model2_test_results(
                fitted_params_patients_model2, N_max_test,
                M_max_test, tx_test, t_test, x1_test,
                x2_test)

        else:
            fitted_params_patients_model3 = pd.read_csv(path + 'model3/fitted_params_patients_' + str(t) + '.csv')
            fitted_params_patients_model4 = pd.read_csv(path + 'model4/fitted_params_patients_' + str(t) + '.csv')
            samples_y_test_model1, samples_y1_test_model1, samples_y2_test_model1 = model3_test_results(
                fitted_params_patients_model3, N_max_test,
                M_max_test, tx_test, t_test, x1_test,
                x2_test)
            samples_y_test_model2, samples_y1_test_model2, samples_y2_test_model2 = model3_test_results(
                fitted_params_patients_model4, N_max_test,
                M_max_test, tx_test, t_test, x1_test,
                x2_test)

        for idx in range(P):
            patient_id = patients[idx]
            df = df_sliced_test[df_sliced_test['id'] == patient_id]
            df_ys = df[~df['y'].isna()]

            trend = [trend_p[idx]] * N_test[idx]
            meals1_model1 = samples_y1_test_model1.iloc[idx, :N_test[idx]]
            meals2_model1 = samples_y2_test_model1.iloc[idx, :N_test[idx]]
            overall_glucose_model1 = samples_y_test_model1.iloc[idx, :N_test[idx]]
            meals1_model2 = samples_y1_test_model2.iloc[idx, :N_test[idx]]
            meals2_model2 = samples_y2_test_model2.iloc[idx, :N_test[idx]]
            overall_glucose_model2 = samples_y_test_model2.iloc[idx, :N_test[idx]]

            # Metrics calculation
            if df_ys['y'].size == 0 or overall_glucose_model1.size == 0 or overall_glucose_model2.size == 0:
                break

            # Plotting
            fig, axs = plt.subplots(3, 1, figsize=(7.5, 5.5), dpi=300, sharex=True)
            fig.tight_layout(pad=2.5)
            plt.xlim(df_ys['t'].min(), df_ys['t'].max())

            axs[0].plot(df_ys['t'], df_ys['y'], 'x', ms=4, label='True observations', c='grey')
            axs[0].plot(df_ys['t'], meals1_model1, c="indigo", linewidth=1.5, label='Posterior mean for carbs')
            axs[0].plot(df_ys['t'], meals2_model1, c="plum", linewidth=1.5, label='Posterior mean for fat')
            axs[0].plot(df_ys['t'], trend, c="grey", linewidth=1.5, label='Baseline')
            axs[0].plot(df_ys['t'], overall_glucose_model1, c="navy", linewidth=1.5, label='Fitted glucose curve')
            axs[0].set(xlabel="Time (hours)", ylabel="Glucose (mmol/l)")
            axs[0].set_title('P-IDR single-patient glucose response to carbs and fat.')
            axs[0].legend(loc='upper right')

            axs[1].plot(df_ys['t'], df_ys['y'], 'x', ms=4, label='True observations', c='grey')
            axs[1].plot(df_ys['t'], meals1_model2, c="indigo", linewidth=1.5, label='Posterior mean for carbs')
            axs[1].plot(df_ys['t'], meals2_model2, c="plum", linewidth=1.5, label='Posterior mean for fat')
            axs[1].plot(df_ys['t'], trend, c="grey", linewidth=1.5, label='Baseline')
            axs[1].plot(df_ys['t'], overall_glucose_model2, c="navy", linewidth=1.5, label='Fitted glucose curve')
            axs[1].set(xlabel="Time (hours)", ylabel="Glucose (mmol/l)")
            axs[1].set_title('P-DR single-patient glucose response to carbs and fat.')
            axs[1].legend(loc='upper right')

            axs[2].bar(df['t'], df['STARCH'] + df['SUGAR'], color='indigo', width=0.3, label="Carbs")
            axs[2].bar(df['t'], df['FAT'], bottom=df['STARCH'] + df['SUGAR'], color='plum', width=0.3, label='Fat')
            axs[2].set(xlabel="Time (hours)", ylabel="Stacked meals (g)")
            axs[2].set_title('True meals, eaten after operation.')
            axs[2].set_axisbelow(True)
            axs[2].grid(which='major', color='#DDDDDD', linewidth=0.8)
            axs[2].legend(loc='upper right')

            patient_id = patient_id.partition('_')[0]
            repo = path + models + '/id' + patient_id
            os.makedirs(repo, exist_ok=True)
            plt.savefig(repo + '/predictions_test_task_' + models + '.pdf')


if __name__ == "__main__":
    plt.rcParams.update({
        "font.family": "Times New Roman",  # use serif/main font for text elements
        "font.size": "8",
        "text.usetex": True,  # use inline math for ticks
        "pgf.rcfonts": False  # don't setup fonts from rc parameters
    })

    args = parser.parse_args()
    os.chdir("../../../")

    # Downloading processed data
    path_processed = args.processed_data
    patients = np.loadtxt(path_processed + 'patients.txt', dtype=str)
    P = len(patients)
    trend_p = np.loadtxt(path_processed + 'trend_p.txt')
    N = np.loadtxt(path_processed + 'N.txt', dtype=int)
    df_sliced = pd.read_csv(path_processed + 'df_sliced.csv')
    df_sliced_test = pd.read_csv(path_processed + 'df_sliced_test.csv')
    N_test = np.loadtxt(path_processed + 'N_test.txt', dtype=int)
    M_test = np.loadtxt(path_processed + 'M_test.txt', dtype=int)
    tx_test = np.loadtxt(path_processed + 'tx_test.txt')
    t_test = np.loadtxt(path_processed + 't_test.txt')
    x1_test = np.loadtxt(path_processed + 'x1_test.txt')
    x2_test = np.loadtxt(path_processed + 'x2_test.txt')

    # # # Computation of results for training data
    metrics_train, rmse_conf_int = analyse_train_results(args.results_data, P,
                                                         patients, df_sliced, trend_p, N, args.tasks)

    # Computation of results for testing data
    metrics_test = analyse_test_results(args.results_data, P, patients,
                                        df_sliced_test, trend_p, N_test, M_test,
                                        args.model, args.tasks)

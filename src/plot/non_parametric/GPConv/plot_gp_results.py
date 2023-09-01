import os
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score
import math
import matplotlib as mpl
import tensorflow as tf


def plot_predictions(data, args, ids, f_means, f_vars, metrics, vars_learnt, time='train', plot_var=True):
    """ Plot predictions for the whole data.
    Parameters:
    data (tuple): tuple of lists of glucose times, glucose values, meals times and values
    ids (list): list of patients' ids
    f_means (list): list of predicted means
    f_vars (list): list of predicted variances
    args (dict): contains input arguments
    metrics (dict): dictionary to store prediction results
    plot_var (boolean): if to plot variance
    """
    plt.rcParams.update({
        "font.family": "Times New Roman",  # use serif/main font for text elements
        "font.size": "16",
        "text.usetex": True,  # use inline math for ticks
        "pgf.rcfonts": False,  # don't setup fonts from rc parameters
        "axes.titlesize": "16",
        "axes.labelsize": "16"
    })
    offset = 0
    x, y, meals = data

    path = args.results_data + '/' + args.meal_type + '/'
    os.makedirs(path, exist_ok=True)

    for i, _ in enumerate(x):
        glucose_len = x[i].shape[0]
        f_means_i = [ff[offset:offset + glucose_len] for ff in f_means]
        f_vars_i = [ff[offset:offset + glucose_len] for ff in f_vars]

        # Metrics calculation
        rmse = math.sqrt(mse(y[i], f_means_i[2]))
        mae_score = mae(y[i], f_means_i[2])
        r2 = r2_score(y[i], f_means_i[2])
        M1 = np.var(f_means_i[0]) / np.var(y[i])
        M2 = np.var(f_means_i[2]) / np.var(y[i]) - M1
        nll = 0.5 * (np.log(2.0 * np.pi) + np.log(vars_learnt[i]) + ((y[i] - f_means_i[2])**2) / vars_learnt[i])
        nll = np.mean(nll)

        # Appending metrics to the metrics dictionary
        metrics['RMSE'].append(rmse)
        metrics['M2'].append(M2)
        metrics['MAE'].append(mae_score)
        metrics['NLL'].append(nll)

        if args.meal_type == 'carbs':
            plot_patient_predictions(ids[i], rmse, x[i], y[i], meals[i], f_means_i, f_vars_i,
                                     ['Baseline', 'Carbs',  'Fitted glucose'],
                                     ['grey',  'darkmagenta', 'royalblue'], path, time, args, plot_var=plot_var)
        elif args.meal_type == 'fat':
            plot_patient_predictions(ids[i], rmse, x[i], y[i], meals[i], f_means_i, f_vars_i,
                                     ['Baseline',  'Fat', 'Fitted glucose'],
                                     ['grey',  'orange', 'royalblue'], path, time, args, plot_var=plot_var)

        offset += glucose_len

    return metrics


def plot_patient_predictions(idx, rmse, x_p, y_p, meals_p, f_means_p, f_vars_p, f_labels, f_colors, path, time='train',
                             args=None, plot_var=True):
    """ Plot predictions for concrete patient.
    Parameters:
    idx (int): patient index
    rmse (float): RMSE
    x_p (np.array): glucose data times for concrete patient
    y_p (np.array): glucose data values for concrete patient
    f_means_p (list): predicted means
    f_vars_p (list): predicted variances
    f_labels (list): list of labels
    f_colors (list): list of colours
    path (str): path to the folder
    plot_var (boolean): if to plot variance
    """
    fig, axs = plt.subplots(2, 1, figsize=(8.0, 4.0), height_ratios=[3,1], dpi=300, sharex=True)
    mpl.rcParams["figure.autolayout"] = True
    plt.xlim(x_p.min(), x_p.max())

    for fm, fv, col, lbl in zip(f_means_p, f_vars_p, f_colors, f_labels):
        if lbl != "Carbs":
            plot_gp_pred(axs[0], x_p, fm, fv, color=col, label=lbl, plot_var=plot_var)

    axs[0].plot(x_p, y_p, 'kx', ms=6, alpha=0.5, label='True observations')
    axs[0].legend(loc='upper right', fontsize=10)
    axs[0].set(ylabel="Glucose (mmol/l)")
    axs[0].set_title('GP-Conv single-patient glucose response to meals.', weight='bold')

    axs[1].bar(meals_p[:, 0], meals_p[:, 1], color='darkmagenta', width=0.3, label="Carbs")
    axs[1].bar(meals_p[:, 0], meals_p[:, 2], bottom=meals_p[:, 1], color='orange', width=0.3, label='Fat')
    axs[1].set(xlabel="Time (hours)", ylabel="Stacked \n meals (g)")
    axs[1].grid(which='major', color='#DDDDDD', linewidth=0.8)
    axs[1].legend(loc='upper right', fontsize=10)

    path = path + 'id' + str(idx) + '/'
    os.makedirs(path, exist_ok=True)
    plt.savefig(path + 'predictions_' + time + '.pdf')
    plt.close()


def plot_gp_pred(ax1, x_p, f_mean, f_var, color='blue', label='Predicted', lw=2.0, linestyle='solid', plot_var=True):
    """ Plot GP predictions intervals for concrete patient.
    Parameters:
    x_p (np.array): glucose data for concrete patient
    f_mean (tf.Tensor): predicted mean
    f_var (tf.Tensor): predicted variance
    color (str): color
    label (str): label
    plot_var (boolean): if to plot variance
    Returns:
    line_gp (plt.lines.Line2D): plotted line
    """
    ax1.plot(x_p, f_mean, color, lw=lw, label=label, linestyle=linestyle, zorder=2)
    if plot_var:
        ax1.fill_between(
            x_p[:, 0],
            f_mean[:, 0] - 1.96 * np.sqrt(f_var[:, 0]),
            f_mean[:, 0] + 1.96 * np.sqrt(f_var[:, 0]),
            color=color,
            alpha=0.2,
        )
    else:
        ax1.fill_between(
            x_p[:, 0],
            f_mean[:, 0],
            color=color,
            alpha=0.2,
        )

########## Plotting one aggregated meal per patient for one type of meal##########
def plot_predictions_meal(data, args, ids, f_means, f_vars, time='train', plot_var=True):
    """ Plot predictions for the whole data.
    Parameters:
    data (tuple): tuple of lists of glucose times, glucose values, meals times and values
    ids (list): list of patients' ids
    f_means (list): list of predicted means
    f_vars (list): list of predicted variances
    args (dict): contains input arguments
    metrics (dict): dictionary to store prediction results
    plot_var (boolean): if to plot variance
    """

    offset = 0
    x, meals = data

    path = args.results_data_meal + '/' + args.meal_type + '/'
    os.makedirs(path, exist_ok=True)

    for i, _ in enumerate(x):
        glucose_len = x[i].shape[0]
        f_means_i = [ff[offset:offset + glucose_len] for ff in f_means]
        f_vars_i = [ff[offset:offset + glucose_len] for ff in f_vars]

        if args.meal_type == 'carbs':
            plot_patient_predictions_meal(ids[i], x[i], meals[i], f_means_i, f_vars_i,
                                     ['Baseline', 'Carbs',  'Fitted glucose'],
                                     ['grey', 'indigo',  'navy'], path, time, args, plot_var=plot_var)
        elif args.meal_type == 'fat':
            plot_patient_predictions_meal(ids[i], x[i], meals[i], f_means_i, f_vars_i,
                                     ['Baseline',  'Fat', 'Fitted glucose'],
                                     ['grey',  'plum', 'navy'], path, time, args, plot_var=plot_var)

        offset += glucose_len



def plot_patient_predictions_meal(idx, x_p, meals_p, f_means_p, f_vars_p, f_labels, f_colors, path,
                                  time='train', args=None, plot_var=True):
    """ Plot predictions for concrete patient.
    Parameters:
    idx (int): patient index
    rmse (float): RMSE
    x_p (np.array): glucose data times for concrete patient
    y_p (np.array): glucose data values for concrete patient
    f_means_p (list): predicted means
    f_vars_p (list): predicted variances
    f_labels (list): list of labels
    f_colors (list): list of colours
    path (str): path to the folder
    plot_var (boolean): if to plot variance
    """
    plt.rcParams['axes.labelsize'] = 13
    fig, axs = plt.subplots(figsize=(20, 10))
    mpl.rcParams["figure.autolayout"] = True

    for fm, fv, col, lbl in zip(f_means_p, f_vars_p, f_colors, f_labels):
            plot_gp_pred(axs, x_p, fm, fv, color=col, label=lbl, plot_var=plot_var, lw=5.0)

    axs.legend(loc='upper right')
    axs.set(ylabel="Glucose (mmol/l)", xlabel="Time (hours)")

    axs2 = axs.twinx()
    if args.meal_type == 'carbs':
        axs2.bar(meals_p[:, 0], meals_p[:, 1], color=f_colors[1], width=0.05, label="Carbs")
    elif args.meal_type == 'fat':
        axs2.bar(meals_p[:, 0], meals_p[:, 2], color=f_colors[1], width=0.05, label="Fat")
    axs2.set_ylabel(ylabel="Stacked meals (g)")
    axs2.set_ylim(0, 50)
    axs2.legend()

    if args.period == 'operation':
        plt.title('Glucose response to '+args.meal_type+', eaten after operation.', fontsize=13)
    elif args.period == 'baseline':
        plt.title('Glucose response to '+args.meal_type+', eaten before operation.', fontsize=13)

    path = path + 'id' + str(idx) + '/'
    os.makedirs(path, exist_ok=True)
    plt.savefig(path + 'predictions_' + time + '.pdf')
    plt.close()

########## Plotting one aggregated meal per patient for full model##########
def plot_predictions_meal_full(data, args, ids, f_means, f_vars, metrics, time='train', plot_var=True):
    """ Plot predictions for the whole data.
    Parameters:
    data (tuple): tuple of lists of glucose times, glucose values, meals times and values
    ids (list): list of patients' ids
    f_means (list): list of predicted means
    f_vars (list): list of predicted variances
    args (dict): contains input arguments
    metrics (dict): dictionary to store prediction results
    plot_var (boolean): if to plot variance
    """

    offset = 0
    x, meals = data

    path = args.results_data_meal + '/' + args.meal_type + '/'
    os.makedirs(path, exist_ok=True)

    for i, _ in enumerate(x):
        glucose_len = x[i].shape[0]
        f_means_i = [ff[offset:offset + glucose_len] for ff in f_means]
        f_vars_i = [ff[offset:offset + glucose_len] for ff in f_vars]

        plot_patient_predictions_meal_full(ids[i], x[i], meals[i], f_means_i, f_vars_i,
                                      ['Baseline', 'Carbs', 'Fat', 'Fitted glucose'],
                                      ['grey', 'indigo', 'plum', 'navy'], path, time, args, plot_var=plot_var)
        offset += glucose_len

    return metrics


def plot_patient_predictions_meal_full(idx, x_p, meals_p, f_means_p, f_vars_p, f_labels, f_colors, path,
                                  time='train', args=None, plot_var=True):
    """ Plot predictions for concrete patient.
    Parameters:
    idx (int): patient index
    rmse (float): RMSE
    x_p (np.array): glucose data times for concrete patient
    y_p (np.array): glucose data values for concrete patient
    f_means_p (list): predicted means
    f_vars_p (list): predicted variances
    f_labels (list): list of labels
    f_colors (list): list of colours
    path (str): path to the folder
    plot_var (boolean): if to plot variance
    """
    plt.rcParams['axes.labelsize'] = 13
    fig, axs = plt.subplots(figsize=(20, 10))
    mpl.rcParams["figure.autolayout"] = True


    for fm, fv, col, lbl in zip(f_means_p, f_vars_p, f_colors, f_labels):
        plot_gp_pred(axs, x_p, fm, fv, color=col, label=lbl, plot_var=plot_var, lw=5.0)

    axs.legend(loc='upper right')
    axs.set(ylabel="Glucose (mmol/l)", xlabel="Time (hours)")

    axs2 = axs.twinx()
    axs2.bar(meals_p[:, 0], meals_p[:, 1], color='indigo', width=0.05, label="Carbs")
    axs2.bar(meals_p[:, 0], meals_p[:, 2], bottom=meals_p[:, 1], color='plum', width=0.05, label='Fat')
    axs2.set_ylabel(ylabel="Stacked meals (g)")
    axs2.set_ylim(0,50)
    axs2.legend()

    if args.period == 'operation':
        plt.title('Glucose response to carbs and fat, eaten after operation.', fontsize=13)
    elif args.period == 'baseline':
        plt.title('Glucose response to carbs and fat, eaten before operation.', fontsize=13)

    path = path + 'id' + str(idx) + '/'
    os.makedirs(path, exist_ok=True)
    plt.savefig(path + 'predictions_' + time + '_full.pdf')
    plt.close()


########## Plotting meal predictions per patient for convolved model on two different datasets (with and without treatment)##########
def plot_predictions_conv(data, args, ids, f_means, f_vars, time='train', full=True, plot_var=True):
    plt.rcParams.update({
        "font.family": "Times New Roman",  # use serif/main font for text elements
        "font.size": "16",
        "text.usetex": True,  # use inline math for ticks
        "pgf.rcfonts": False,  # don't setup fonts from rc parameters
        "axes.titlesize": "16",
        "axes.labelsize": "16"
    })

    offset = 0
    x, y, meals_wo_fat, meals = data

    path_arrays = args.created_arrays_path
    os.makedirs(path_arrays, exist_ok=True)

    for i, _ in enumerate(x):
        glucose_len = x[i].shape[0]
        f_means_i = [ff[offset:offset + glucose_len] for ff in f_means]
        f_vars_i = [ff[offset:offset + glucose_len] for ff in f_vars]

        if full:
            path = args.results_data + '/' + args.meal_type + '/'
            os.makedirs(path, exist_ok=True)
            plot_patient_predictions_conv(ids[i], x[i], y[i], meals[i], f_means_i, f_vars_i,
                                     ['Baseline', 'Fitted glucose without fat', 'Fitted glucose with fat', 'Glucose response to meals with fat', 'Glucose response to meals without fat'],
                                     ['grey', 'dodgerblue', 'mediumblue', 'orange', 'darkmagenta'],
                                     ['solid', 'solid', 'solid', 'dotted', 'dotted'], path, time, args, plot_var=plot_var)

            # Save arrays with results data
            file = path_arrays + ids[i] + '_' + time + '_'+'full.npz'
            np.savez(file, baseline=f_means_i[0], fitted_glucose=f_means_i[2], fitted_glucose_wofat=f_means_i[1], glucose=f_means_i[3],glucose_wofat=f_means_i[4],
                     baseline_var=f_vars_i[0], fitted_glucose_var=f_vars_i[2], fitted_glucose_wofat_var=f_vars_i[1], glucose_var=f_vars_i[3],glucose_wofat_var=f_vars_i[4])
        else:
            path = args.results_data_meal + '/' + args.meal_type + '/'
            os.makedirs(path, exist_ok=True)
            plot_patient_predictions_conv_meal(ids[i], x[i], meals[i], f_means_i, f_vars_i,
                                          ['Baseline', 'Fitted glucose with fat', 'Fitted glucose without fat',
                                           'Response \n without fat', 'Response \n with fat'],
                                          ['orange', 'darkmagenta', 'grey', 'dodgerblue', 'mediumblue'],
                                          ['solid', 'solid', 'dashed', 'solid', 'solid'],path, time, args,
                                          plot_var=plot_var)

            # Save arrays with results data
            file = path_arrays + ids[i] + '_' + time + '_'+'onemeal.npz'
            np.savez(file, baseline=f_means_i[0], fitted_glucose=f_means_i[1], fitted_glucose_wofat=f_means_i[2], glucose=f_means_i[3],glucose_wofat=f_means_i[4],
                     baseline_var=f_vars_i[0], fitted_glucose_var=f_vars_i[1], fitted_glucose_wofat_var=f_vars_i[2], glucose_var=f_vars_i[3],glucose_wofat_var=f_vars_i[4])

        offset += glucose_len

def plot_patient_predictions_conv(idx, x_p, y_p, meals_p, f_means_p, f_vars_p, f_labels, f_colors, f_linestyle, path, time='train',
                             args=None, plot_var=True):
    fig, axs = plt.subplots(2, 1, figsize=(8.0, 4.0), height_ratios=[3,1], dpi=300, sharex=True)
    plt.xlim(x_p.min(), x_p.max())

    for fm, fv, col, lbl, ls in zip(f_means_p[:3], f_vars_p[:3], f_colors[:3], f_labels[:3], f_linestyle[:3]):
            plot_gp_pred(axs[0], x_p, fm, fv, color=col, label=lbl, linestyle=ls, plot_var=plot_var)
        # if lbl == 'fitted glucose without fat':
        #     axs[0].fill_between(np.squeeze(x_p), tf.squeeze(f_means_p[0]),tf.squeeze(f_means_p[1]), color='dodgerblue', alpha=0.3)

    axs[0].plot(x_p, y_p, 'x', ms=7, alpha=0.65, label='true observations', c='grey')
    axs[0].legend(loc='upper right', fontsize=8,frameon=False)
    axs[0].set(ylabel="Glucose (mmol/l)")
    axs[0].set_ylim(3,6)
    axs[0].set_title('GP-Conv glucose response to meals with and without fat')

    axs[1].bar(meals_p[:, 0], meals_p[:, 1], color=f_colors[4], width=0.3, label="Carbs")
    axs[1].bar(meals_p[:, 0], meals_p[:, 2], bottom=meals_p[:, 1], color=f_colors[3], width=0.3, label='Fat')
    axs[1].set(xlabel="Time (hours)", ylabel="Stacked \n meals (g)")
    axs[1].grid(which='major', color='#DDDDDD', linewidth=0.8)
    axs[1].legend(loc='upper right', fontsize=8)

    path = path + 'id' + str(idx) + '/'
    os.makedirs(path, exist_ok=True)
    plt.savefig(path + 'predictions_setups_' + time + '.pdf')
    plt.close()

def plot_patient_predictions_conv_meal(idx, x_p, meals_p, f_means_p, f_vars_p, f_labels, f_colors, f_linestyle, path, time='train',
                             args=None, plot_var=True):
    mpl.rcParams["figure.autolayout"] = True
    fig, axs = plt.subplots(figsize=(5, 3), dpi=300)
    plt.xlim(x_p.min(), x_p.max())

    axs.set(ylabel="Stacked meals (g)")
    axs.bar(meals_p[:, 0], meals_p[:, 1], color=f_colors[1], width=0.3, label='Carbs')
    axs.bar(meals_p[:, 0], meals_p[:, 2], bottom=meals_p[:, 1], color=f_colors[0], width=0.3, label='Fat')
    axs.set_ylim(0, 50)
    axs.legend(loc='upper left', fontsize=8)

    axs2 = axs.twinx()
    axs2.set(ylabel="Glucose (mmol/l)")
    plot_vars = [False, True]
    for fm, fv, col, lbl, ls, pv in zip(f_means_p[3:], f_vars_p[3:], f_colors[3:], f_labels[3:], f_linestyle[3:], plot_vars):
        plot_gp_pred(axs2, x_p, fm, fv, color=col, label=lbl, linestyle=ls, plot_var=pv)


    axs2.legend(loc='upper right', fontsize=8)
    axs2.set_xticklabels([])
    plt.title('GP-Conv')

    path = path + 'id' + str(idx) + '/'
    os.makedirs(path, exist_ok=True)
    plt.savefig(path + 'predictions_setups_' + time + '.pdf')
    plt.close()


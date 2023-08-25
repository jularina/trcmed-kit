import os
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
from sklearn.metrics import mean_squared_error as mse
import math


def plot_predictions(data, args, ids, f_means_op, f_vars_op, f_means_b, f_vars_b, time='train', plot_var=True):
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
    x, y, meals = data

    path = args.results_data
    os.makedirs(path, exist_ok=True)

    for i, _ in enumerate(x):
        glucose_len = x[i].shape[0]
        f_means_i_op = [ff[offset:offset + glucose_len] for ff in f_means_op]
        f_vars_i_op = [ff[offset:offset + glucose_len] for ff in f_vars_op]
        f_means_i_b = [ff[offset:offset + glucose_len] for ff in f_means_b]
        f_vars_i_b = [ff[offset:offset + glucose_len] for ff in f_vars_b]

        plot_patient_predictions(ids[i], x[i], y[i], meals[i], f_means_i_op, f_vars_i_op, f_means_i_b, f_vars_i_b,
                     ['baseline_post', 'Carbs post-operation', 'Fat post-operation', 'fitted glucose_post'],
                     ['baseline_pre', 'Carbs pre-operation', 'Fat pre-operation', 'fitted glucose_pre'],
                     ['grey', 'indigo', 'plum', 'navy'],
                     ['grey', 'indigo', 'plum', 'navy'],
                    path, args.period, time, plot_var=plot_var)
        offset += glucose_len


def plot_patient_predictions(idx, x_p, y_p, meals_p, f_means_p_op, f_vars_p_op, f_means_p_b, f_vars_p_b, f_labels_op, f_labels_b, f_colors_op, f_colors_b, path, period, time='train', plot_var=True):
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
    plt.rcParams['axes.labelsize'] = 15
    fig, axs = plt.subplots(3, 1, figsize=(20, 10), sharex=True)
    fig.suptitle('Glucose response prediction for patient {}'.format(idx), weight='bold', fontsize=15)

    # Second type of plot
    plot_gp_pred(axs[0], x_p, f_means_p_op[1], f_vars_p_op[1], color=f_colors_op[1], label=f_labels_op[1], lw=3, facecolor=f_colors_op[1], alpha=0.2, plot_var=plot_var)
    plot_gp_pred(axs[0], x_p, f_means_p_b[1], f_vars_p_b[1], color=f_colors_b[1], label=f_labels_b[1], linestyle='dashed', hatch = 'oo', plot_var=plot_var)
    axs[0].legend(loc='upper right')
    axs[0].set(ylabel="Glucose (mmol/l)")
    axs[0].title.set_text('Glucose response to carbs.')

    plot_gp_pred(axs[1], x_p, f_means_p_op[2], f_vars_p_op[2], color=f_colors_op[2], label=f_labels_op[2], lw=3, facecolor=f_colors_op[2], alpha=0.2, plot_var=plot_var)
    plot_gp_pred(axs[1], x_p, f_means_p_b[2], f_vars_p_b[2], color=f_colors_b[2], label=f_labels_b[2], linestyle='dashed', hatch = 'oo', plot_var=plot_var)
    axs[1].legend(loc='upper right')
    axs[1].set(ylabel="Glucose (mmol/l)")
    axs[1].title.set_text('Glucose response to fat.')

    axs[2].bar(meals_p[:,0], meals_p[:,1], color='indigo', width=0.3, label="Carbs")
    axs[2].bar(meals_p[:,0], meals_p[:,2], bottom=meals_p[:,1], color='plum', width=0.3, label='Fat')
    axs[2].set(xlabel="Time (hours)", ylabel="Stacked meals (g)")
    if period == 'operation':
        axs[2].title.set_text('True meals, eaten after operation.')
    elif period == 'baseline':
        axs[2].title.set_text('True meals, eaten before operation.')
    axs[2].legend()

    plt.subplots_adjust(top=0.93)

    path = path + 'id' + str(idx) + '/'
    os.makedirs(path, exist_ok=True)
    plt.savefig(path+'predictions_'+ period +'_'+time+'.jpg')
    plt.close()


def plot_gp_pred(ax1, x_p, f_mean, f_var, color='blue', label='Predicted', lw=2, linestyle='solid', hatch='', facecolor='none', alpha=1.0, plot_var=True):
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
    ax1.plot(x_p, f_mean, color, lw=lw, label=label, zorder=2, linestyle=linestyle)
    if plot_var:
        ax1.fill_between(
            x_p[:, 0],
            f_mean[:, 0],
            edgecolor=color,
            facecolor = facecolor,
            alpha=alpha,
            hatch=hatch
        )

def plot_predictions_meal(data, args, ids, f_means_op, f_vars_op, f_means_b, f_vars_b, time='train', plot_var=True):
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

    path = args.results_data + args.period + '/'
    os.makedirs(path, exist_ok=True)

    for i, _ in enumerate(x):
        glucose_len = x[i].shape[0]
        f_means_i_op = [ff[offset:offset + glucose_len] for ff in f_means_op]
        f_vars_i_op = [ff[offset:offset + glucose_len] for ff in f_vars_op]
        f_means_i_b = [ff[offset:offset + glucose_len] for ff in f_means_b]
        f_vars_i_b = [ff[offset:offset + glucose_len] for ff in f_vars_b]

        plot_patient_predictions_meal(ids[i], x[i], meals[i], f_means_i_op, f_vars_i_op, f_means_i_b, f_vars_i_b,
                     ['baseline_post', 'Carbs post-operation', 'Fat post-operation', 'fitted glucose_post'],
                     ['baseline_pre', 'Carbs pre-operation', 'Fat pre-operation', 'fitted glucose_pre'],
                     ['grey', 'indigo', 'plum', 'navy'],
                     ['grey', 'indigo', 'plum', 'navy'],
                    path, args.period, time, plot_var=plot_var)
        offset += glucose_len

def plot_patient_predictions_meal(idx, x_p, meals_p, f_means_p_op, f_vars_p_op, f_means_p_b, f_vars_p_b, f_labels_op, f_labels_b, f_colors_op, f_colors_b, path, period, time='train', plot_var=True):
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
    plt.rcParams['axes.labelsize'] = 15
    mpl.rcParams["figure.autolayout"] = True
    fig, axs = plt.subplots(1, 2, figsize=(30, 10), sharex=True)
    fig.suptitle('Glucose response to carbs and fat for patient {}'.format(idx), weight='bold', fontsize=15)

    plot_gp_pred(axs[0], x_p, f_means_p_op[1], f_vars_p_op[1], color=f_colors_op[1], label=f_labels_op[1], lw=3, facecolor=f_colors_op[1], alpha=0.2, plot_var=plot_var)
    plot_gp_pred(axs[0], x_p, f_means_p_b[1], f_vars_p_b[1], color=f_colors_b[1], label=f_labels_b[1], linestyle='dashed', hatch = 'oo', plot_var=plot_var)
    axs[0].legend(loc='upper left')
    axs[0].set(ylabel="Glucose (mmol/l)", xlabel="Time (hours)")
    axs[0].title.set_text('Glucose response to carbs.')
    axs02 = axs[0].twinx()
    axs02.bar(meals_p[:, 0], meals_p[:, 1], color='indigo', width=0.3, label="Carbs")
    axs02.set_ylabel(ylabel="Stacked meals (g)")
    axs02.set_ylim(0, 50)
    axs02.legend(loc='upper right')

    plot_gp_pred(axs[1], x_p, f_means_p_op[2], f_vars_p_op[2], color=f_colors_op[2], label=f_labels_op[2], lw=3, facecolor=f_colors_op[2], alpha=0.2, plot_var=plot_var)
    plot_gp_pred(axs[1], x_p, f_means_p_b[2], f_vars_p_b[2], color=f_colors_b[2], label=f_labels_b[2], linestyle='dashed', hatch = 'oo', plot_var=plot_var)
    axs[1].legend(loc='upper left')
    axs[1].set(ylabel="Glucose (mmol/l)", xlabel="Time (hours)")
    axs[1].title.set_text('Glucose response to fat.')
    axs12 = axs[1].twinx()
    axs12.bar(meals_p[:, 0], meals_p[:, 2], color='plum', width=0.3, label="Fat")
    axs12.set_ylabel(ylabel="Stacked meals (g)")
    axs12.set_ylim(0,50)
    axs12.legend(loc='upper right')

    path = path + 'id' + str(idx) + '/'
    os.makedirs(path, exist_ok=True)
    plt.savefig(path+'predictions_'+ period +'_'+time+'.jpg')
    plt.close()
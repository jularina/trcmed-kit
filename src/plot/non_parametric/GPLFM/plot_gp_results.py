import os
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score
import math
import matplotlib as mpl


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

    path = args.results_data + '/'
    os.makedirs(path, exist_ok=True)

    for i, _ in enumerate(x):
        glucose_len = x[i].shape[0]
        f_means_i = [ff[offset:offset + glucose_len] for ff in f_means]
        f_vars_i = [ff[offset:offset + glucose_len] for ff in f_vars]

        # Metrics calculation
        rmse = math.sqrt(mse(y[i], f_means_i[3]))
        mae_score = mae(y[i], f_means_i[3])
        r2 = r2_score(y[i], f_means_i[3])
        M1 = np.var(f_means_i[0]) / np.var(y[i])
        M2 = np.var(f_means_i[3]) / np.var(y[i]) - M1
        nll = 0.5 * (np.log(2.0 * np.pi) + np.log(vars_learnt[i]) + ((y[i] - f_means_i[3])**2) / vars_learnt[i])
        nll = np.mean(nll)

        # Appending metrics to the metrics dictionary
        metrics['RMSE'].append(rmse)
        metrics['M2'].append(M2)
        metrics['MAE'].append(mae_score)
        metrics['NLL'].append(nll)

        plot_patient_predictions(ids[i], rmse, x[i], y[i], meals[i], f_means_i, f_vars_i,
                                 ['baseline', 'response to carbs', 'response to fat', 'Fitted glucose'],
                                 ['grey', 'darkmagenta', 'orange', 'royalblue'], path, time, args, plot_var=plot_var)
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
    mpl.rcParams["figure.autolayout"] = True
    fig, axs = plt.subplots(2, 1, figsize=(8.0, 4.0), height_ratios=[3,1], dpi=300, sharex=True)
    plt.xlim(x_p.min(), x_p.max())

    for fm, fv, col, lbl in zip(f_means_p, f_vars_p, f_colors, f_labels):
        plot_gp_pred(axs[0], x_p, fm, fv, color=col, label=lbl, plot_var=False)

    axs[0].plot(x_p, y_p, 'x', ms=7, alpha=0.65, label='true observations', c='grey')
    axs[0].legend(loc='upper right',fontsize=8, frameon=False)
    axs[0].set(ylabel="Glucose (mmol/l)")
    axs[0].set_ylim(4, 7)
    axs[0].set_title('GP-LFM glucose response to carbs and fat')

    axs[1].bar(meals_p[:, 0], meals_p[:, 1], color='darkmagenta', width=0.3, label="Carbs")
    axs[1].bar(meals_p[:, 0], meals_p[:, 2], bottom=meals_p[:, 1], color='orange', width=0.3, label='Fat')
    axs[1].set(xlabel="Time (hours)", ylabel="Stacked \n meals (g)")
    axs[1].grid(which='major', color='#DDDDDD', linewidth=0.8)
    axs[1].legend(loc='upper right',fontsize=10)

    path = path + 'id' + str(idx) + '/'
    os.makedirs(path, exist_ok=True)
    plt.savefig(path + 'predictions_' + time + '.pdf')
    plt.close()


    # plt.figure(figsize=(8.0,3.0))
    # for fm, fv, col, lbl in zip(f_means_p, f_vars_p, f_colors, f_labels):
    #     plt.plot(x_p, fm, color=col, lw=1.5, label=lbl, linestyle='solid', zorder=2)
    #     plt.fill_between(
    #         x_p[:, 0],
    #         fm[:, 0] - 1.96 * np.sqrt(fv[:, 0]),
    #         fm[:, 0] + 1.96 * np.sqrt(fv[:, 0]),
    #         color=col,
    #         alpha=0.2,
    #     )
    # plt.plot(x_p, y_p, 'x', ms=8, alpha=0.5, label='true observations', c='grey')
    # plt.legend(loc='upper right',fontsize=10)
    # plt.xlabel("Time (hours)")
    # plt.ylabel("Glucose (mmol/l)")
    # plt.title('GP-LFM glucose response to carbs and fat.')
    # ax = plt.gca()
    # ax.axes.xaxis.set_ticklabels([])
    # plt.savefig(path + 'predictionsfull_' + time + '.pdf')
    # plt.close()
    #
    # plt.figure(figsize=(8.0, 2.0))
    # plt.bar(meals_p[:, 0], meals_p[:, 1], color='darkmagenta', width=0.3, label="Carbs")
    # plt.bar(meals_p[:, 0], meals_p[:, 2], bottom=meals_p[:, 1], color='orange', width=0.3, label='Fat')
    # plt.xlabel("Time (hours)")
    # plt.ylabel("Stacked \n meals (g)")
    # plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
    # plt.legend(loc='upper right',fontsize=10)
    # plt.title('True meals, eaten after operation.')
    # plt.savefig(path + 'predictionsavg_' + time + '.pdf')
    # plt.close()

def plot_gp_pred(ax1, x_p, f_mean, f_var, color='blue', label='Predicted', lw=2.0, ls='solid', plot_var=True):
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
    ax1.plot(x_p, f_mean, color, lw=lw, label=label, linestyle=ls, zorder=2)
    if plot_var:
        ax1.fill_between(
            x_p[:, 0],
            f_mean[:, 0] - 1.96 * np.sqrt(f_var[:, 0]),
            f_mean[:, 0] + 1.96 * np.sqrt(f_var[:, 0]),
            color=color,
            alpha=0.2,
        )


########## Plotting one aggregated meal per patient ##########
def plot_predictions_meal(data, args, ids, f_means, f_vars, metrics, time='train', plot_var=True):
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
    x, meals = data

    path = args.results_data_meal + '/'
    os.makedirs(path, exist_ok=True)

    for i, _ in enumerate(x):
        glucose_len = x[i].shape[0]
        f_means_i = [ff[offset:offset + glucose_len] for ff in f_means]
        f_vars_i = [ff[offset:offset + glucose_len] for ff in f_vars]

        plot_patient_predictions_meal(ids[i], x[i], meals[i], f_means_i, f_vars_i,
                                      ['baseline', 'response \n to carbs', 'response \n to fat', 'fitted glucose'],
                                      ['grey', 'darkmagenta', 'orange', 'blue'], path, time, args, plot_var=plot_var)
        offset += glucose_len

    return metrics


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
    fig, axs = plt.subplots(figsize=(5,3),dpi=300)
    plt.xlim(x_p.min(), x_p.max())
    mpl.rcParams["figure.autolayout"] = True

    axs.bar(meals_p[:, 0], meals_p[:, 1], color='darkmagenta', width=0.3, label="Carbs")
    axs.bar(meals_p[:, 0], meals_p[:, 2], bottom=meals_p[:, 1], color='orange', width=0.3, label='Fat')
    axs.set(ylabel="Stacked meals (g)", xlabel="Time (hours)")
    axs.set_ylim(0,50)

    axs2 = axs.twinx()
    for fm, fv, col, lbl in zip(f_means_p, f_vars_p, f_colors, f_labels):
        if lbl != 'baseline' and lbl != 'fitted glucose':
            plot_gp_pred(axs2, x_p, fm, fv, color=col, label=lbl, plot_var=plot_var, lw=1.5)

    axs2.set(ylabel="Glucose (mmol/l)", xlabel="Time (hours)")
    axs2.legend(loc='upper right', fontsize=8)

    plt.title('GP-LFM')

    path = path + 'id' + str(idx) + '/'
    os.makedirs(path, exist_ok=True)
    plt.savefig(path + 'predictions_' + time + '.pdf')
    plt.close()



#### Plot latent predictions
def plot_latent(data, args, ids, f_means, f_vars, time='train', plot_var=True):
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

    path = args.results_data +  '/'
    os.makedirs(path, exist_ok=True)

    for i, _ in enumerate(x):
        glucose_len = x[i].shape[0]
        f_means_i = [ff[offset:offset + glucose_len] for ff in f_means]
        f_vars_i = [ff[offset:offset + glucose_len] for ff in f_vars]

        plot_patient_predictions_latent(ids[i], x[i], f_means_i, f_vars_i,
                                 ['latent response to carbs', 'response to carbs', 'latent response to fat', 'response to fat'],
                                 ['darkmagenta', 'darkmagenta', 'orange', 'orange'], ['dotted','solid','dotted','solid'], path, time, args, plot_var=plot_var)
        offset += glucose_len

def plot_patient_predictions_latent(idx, x_p, f_means_p, f_vars_p, f_labels, f_colors, f_lines, path, time='train',
                             args=None, plot_var=True):
    mpl.rcParams["figure.autolayout"] = True
    fig, axs = plt.subplots(2, 1, figsize=(6.0, 4.0), dpi=300, sharex=True)
    plt.xlim(x_p.min(), x_p.max())
    for fm, fv, col, lbl, ls in zip(f_means_p[:2], f_vars_p[:2], f_colors[:2], f_labels[:2], f_lines[:2]):
        plot_gp_pred(axs[0], x_p, fm, fv, color=col, label=lbl, ls = ls, plot_var=False)

    axs[0].legend(loc='upper right', fontsize=10)
    axs[0].set_title('Carbs')

    for fm, fv, col, lbl, ls in zip(f_means_p[2:], f_vars_p[2:], f_colors[2:], f_labels[2:], f_lines[2:]):
        plot_gp_pred(axs[1], x_p, fm, fv, color=col, label=lbl, ls = ls, plot_var=False)

    axs[1].legend(loc='upper right', fontsize=10)
    axs[1].set(xlabel="Time (hours)")
    axs[1].set_title('Fat')
    fig.text(0.00, 0.5, 'Glucose (mmol/l)', va='center', rotation='vertical')

    path = path + 'id' + str(idx) + '/'
    os.makedirs(path, exist_ok=True)
    plt.savefig(path + 'latent_predictions_' + time + '.pdf')
    plt.close()

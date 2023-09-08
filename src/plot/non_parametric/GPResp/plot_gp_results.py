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

    path = args.results_data
    os.makedirs(path, exist_ok=True)

    path_arrays = args.created_arrays_path
    os.makedirs(path_arrays, exist_ok=True)

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
        M5 = abs(np.var(np.array(f_means_i[1]) + np.array(f_means_i[2])) - np.var(y[i]))
        nll = 0.5 * (np.log(2.0 * np.pi) + np.log(vars_learnt[i]) + ((y[i] - f_means_i[3])**2) / vars_learnt[i])
        nll = np.mean(nll)

        # Appending metrics to the metrics dictionary
        metrics['RMSE'].append(rmse)
        metrics['M2'].append(M2)
        metrics['MAE'].append(mae_score)
        metrics['NLL'].append(nll)

        if args.cross_val is not True:
            plot_patient_predictions(ids[i], rmse, x[i], y[i], meals[i], f_means_i, f_vars_i,
                                     ['Baseline', 'Carbs', 'Fat', 'Fitted glucose'],
                                     ['grey', 'darkmagenta', 'orange', 'royalblue'], path, time, args, plot_var=plot_var)

            # Save arrays with results data
            file = path_arrays + ids[i] + '_' + time + '.npz'
            np.savez(file, baseline=f_means_i[0], carbs=f_means_i[1], fat=f_means_i[2], fitted_glucose=f_means_i[3],
                     baseline_var=f_vars_i[0], carbs_var=f_vars_i[1], fat_var=f_vars_i[2], fitted_glucose_var=f_vars_i[3])

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
    axs[0].legend(loc='upper right', fontsize=8, frameon=False)
    axs[0].set(ylabel="Glucose (mmol/l)")
    axs[0].set_ylim(4.0,7.0)
    axs[0].set_title('GP-Resp glucose response to carbs and fat.')

    axs[1].bar(meals_p[:, 0], meals_p[:, 1], color='darkmagenta', width=0.3, label="Carbs")
    axs[1].bar(meals_p[:, 0], meals_p[:, 2], bottom=meals_p[:, 1], color='orange', width=0.3, label='Fat')
    axs[1].set(xlabel="Time (hours)", ylabel="Stacked \n meals (g)")
    axs[1].grid(which='major', color='#DDDDDD', linewidth=0.8)
    axs[1].legend(loc='upper right', fontsize=6)

    path = path + 'id' + str(idx) + '/'
    os.makedirs(path, exist_ok=True)
    plt.savefig(path + 'predictions_' + time + '.pdf')
    plt.close()


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
    ax1.plot(x_p, f_mean, color, lw=lw, linestyle=ls, label=label, zorder=2)
    if plot_var:
        ax1.fill_between(
            x_p[:, 0],
            f_mean[:, 0] - 1.96 * np.sqrt(f_var[:, 0]),
            f_mean[:, 0] + 1.96 * np.sqrt(f_var[:, 0]),
            color=color,
            alpha=0.2,
        )


########## Plotting one aggregated meal per patient ##########
def plot_predictions_meal(data, args, ids, f_means, f_vars, metrics, time='train', order='original', plot_var=True):
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
        "font.size": "8",
        "text.usetex": True,  # use inline math for ticks
        "pgf.rcfonts": False  # don't setup fonts from rc parameters
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
                                      ['baseline', 'carbs', 'fat', 'fitted glucose'],
                                      ['grey', 'indigo', 'plum', 'navy'], path, time, order, args, plot_var=plot_var)
        offset += glucose_len

    return metrics


def plot_patient_predictions_meal(idx, x_p, meals_p, f_means_p, f_vars_p, f_labels, f_colors, path,
                                  time='train', order='original', args=None, plot_var=True):
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
    fig, axs = plt.subplots(figsize=(4.0, 3.0))
    mpl.rcParams["figure.autolayout"] = True


    for fm, fv, col, lbl in zip(f_means_p, f_vars_p, f_colors, f_labels):
        if lbl != 'baseline' and lbl != 'fitted glucose':
            plot_gp_pred(axs, x_p, fm, fv, color=col, label=lbl, plot_var=plot_var, lw=1.5)

    axs.legend(loc='upper right', fontsize=5)
    axs.set(ylabel="Glucose (mmol/l)", xlabel="Time (hours)")

    axs2 = axs.twinx()
    axs2.bar(meals_p[:, 0], meals_p[:, 1], color='indigo', width=0.05, label="Carbs")
    axs2.bar(meals_p[:, 0], meals_p[:, 2], bottom=meals_p[:, 1], color='plum', width=0.05, label='Fat')
    axs2.set_ylabel(ylabel="Stacked meals (g)")
    axs2.set_ylim(0,50)
    axs2.legend(loc='upper right', fontsize=5)

    if args.period == 'operation':
        plt.title('NP-SepR single-patient glucose \n response to averaged meals.', fontsize=11)
    elif args.period == 'baseline':
        plt.title('NP-SepR single-patient glucose \n response to averaged meals.', fontsize=11)

    path = path + 'id' + str(idx) + '/'
    os.makedirs(path, exist_ok=True)
    plt.savefig(path + 'predictions_' + time + '_'+order+'.pdf')
    plt.close()


############# Plots for single meal and several data setups #############

def plot_predictions_meal_severalsetups(data, args, ids, f_means, f_vars, plot_var=True):
    plt.rcParams.update({
        "font.family": "Times New Roman",  # use serif/main font for text elements
        "font.size": "8",
        "text.usetex": True,  # use inline math for ticks
        "pgf.rcfonts": False  # don't setup fonts from rc parameters
    })

    offset = 0
    x, meals, meals_same, meals_reverse = data

    path = args.results_data_meal + '/'
    os.makedirs(path, exist_ok=True)

    for i, _ in enumerate(x):
        glucose_len = x[i].shape[0]
        f_means_i = [ff[offset:offset + glucose_len] for ff in f_means]
        f_vars_i = [ff[offset:offset + glucose_len] for ff in f_vars]

        plot_patient_predictions_meal_severalsetups(ids[i], x[i], [meals[i],meals_same[i],meals_reverse[i]], f_means_i, f_vars_i,
                                      ['carbs', 'fat'],
                                      ['carbs ' + r'$>$' + ' fat','carbs = fat','carbs ' + r'$<$' + ' fat'],
                                      ['darkmagenta','orange'],
                                      ['solid', 'dashed','dotted'], path, args, plot_var=plot_var)
        offset += glucose_len

def plot_patient_predictions_meal_severalsetups(idx, x_p, meals_p, f_means_p, f_vars_p, f_labels, setups, f_colors, linestyles, path,
                                  args=None, plot_var=True):

    fig = plt.figure(figsize=(7.5, 3.5), dpi=300)
    ax1 = plt.subplot2grid((2, 6), (0, 0), rowspan=1, colspan=2)
    ax2 = plt.subplot2grid((2, 6), (0, 2), rowspan=1, colspan=2)
    ax3 = plt.subplot2grid((2, 6), (0, 4), rowspan=1, colspan=2)
    axstop = [ax1,ax2,ax3]

    ax4 = plt.subplot2grid((2, 6), (1, 0), rowspan=1, colspan=3)
    ax5 = plt.subplot2grid((2, 6), (1, 3), rowspan=1, colspan=3)


    for i, ax in enumerate(axstop):
        ax.bar(meals_p[i][:, 0], meals_p[i][:, 1], color='darkmagenta', width=0.05, label="Carbs")
        ax.bar(meals_p[i][:, 0], meals_p[i][:, 2], bottom=meals_p[i][:, 1], color='orange', width=0.05, label='Fat')
        ax.set_ylabel(ylabel="Stacked meals (g)")
        ax.set_ylim(0,50)
        ax.legend(loc='upper right',fontsize=6)
        ax2 = ax.twinx()
        for j in range(2):
            plot_gp_pred(ax2, x_p, f_means_p[i*2+j], f_vars_p[i*2+j], color=f_colors[j], label=f_labels[j], plot_var=plot_var, lw=1.5, ls=linestyles[i])

        ax2.set(ylabel="Glucose (mmol/l)", xlabel="Time (hours)")
        ax2.set_title('Glucose response to \n averaged meals wtih '+setups[i]+'.')

    for i in range(3):
        plot_gp_pred(ax4, x_p, f_means_p[i * 2], f_vars_p[i * 2], color=f_colors[0], label=setups[i],
                     plot_var=False, lw=1.5, ls=linestyles[i])
    ax4.legend(loc='upper right',fontsize=6)
    ax4.set(ylabel="Glucose (mmol/l)", xlabel="Time (hours)")
    ax4.set_title('Comparison of carbs response curves.')
    ax4.fill_between(x_p[:, 0], f_means_p[0][:, 0], color=f_colors[0], alpha=0.1)
    ax4.fill_between(x_p[:, 0], f_means_p[2][:, 0], color=f_colors[0], alpha=0.18)
    ax4.fill_between(x_p[:, 0], f_means_p[4][:, 0], fc=f_colors[0], alpha=0.24)

    for i in range(3):
        plot_gp_pred(ax5, x_p, f_means_p[i * 2+1], f_vars_p[i * 2+1], color=f_colors[1], label=setups[i],
                     plot_var=False, lw=1.5, ls=linestyles[i])
    ax5.legend(loc='upper right',fontsize=6)
    ax5.set(xlabel="Time (hours)")
    ax5.set_title('Comparison of fat response curves.')
    ax5.fill_between(x_p[:, 0], f_means_p[1][:, 0], color=f_colors[1], alpha=0.14)
    ax5.fill_between(x_p[:, 0], f_means_p[3][:, 0], f_means_p[1][:, 0], color=f_colors[1], alpha=0.22)
    ax5.fill_between(x_p[:, 0], f_means_p[5][:, 0], f_means_p[3][:, 0], color=f_colors[1], alpha=0.3)

    path = path + 'id' + str(idx) + '/'
    os.makedirs(path, exist_ok=True)
    plt.savefig(path + 'predictions_test_severalsetups.pdf')
    plt.close()
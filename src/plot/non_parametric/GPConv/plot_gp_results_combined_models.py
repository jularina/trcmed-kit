import os
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
from sklearn.metrics import mean_squared_error as mse
import math


def plot_predictions_meal(data, args, ids, f_means, f_vars, f_means_full, f_vars_full, time='train', plot_var=True):
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

    path = args.results_data_meal + args.period + '/' + args.meal_type + '/'
    os.makedirs(path, exist_ok=True)

    for i, _ in enumerate(x):
        glucose_len = x[i].shape[0]
        f_means_i = [ff[offset:offset + glucose_len] for ff in f_means]
        f_vars_i = [ff[offset:offset + glucose_len] for ff in f_vars]
        f_means_i_full = [ff[offset:offset + glucose_len] for ff in f_means_full]
        f_vars_i_full = [ff[offset:offset + glucose_len] for ff in f_vars_full]

        plot_patient_predictions_meal(ids[i], x[i], meals[i], f_means_i, f_vars_i, f_means_i_full, f_vars_i_full,
                     ['Baseline', 'Carbs', 'Fitted glucose_post'],
                     ['Baseline full', 'Carbs, when fat added', 'Fat added', 'Fitted glucose full'],
                     ['grey', 'grey', 'navy'],
                     ['grey', 'indigo', 'hotpink', 'navy'],
                    path, args, time, plot_var=plot_var)
        offset += glucose_len

def plot_patient_predictions_meal(idx, x_p, meals_p, f_means_p, f_vars_p, f_means_p_full, f_vars_p_full, f_labels, f_labels_full, f_colors, f_colors_full, path, args, time='train', plot_var=True):
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
    fig.suptitle('Additive change to the glucose response for {} patient {}'.format(args.meal_type, idx), weight='bold', fontsize=15)

    plot_gp_pred(axs[0], x_p, f_means_p[1], f_vars_p[1], color=f_colors[1], label=f_labels[1], lw=2, facecolor=f_colors[1], alpha=0.2, plot_var=plot_var)
    axs[0].legend(loc='upper left')
    axs[0].set(ylabel="Glucose (mmol/l)", xlabel="Time (hours)")
    axs[0].title.set_text('Glucose response only to {}.'.format(args.meal_type))
    axs02 = axs[0].twinx()
    if args.meal_type == 'carbs':
        axs02.bar(meals_p[:, 0], meals_p[:, 1], color=f_colors[1], width=0.05, label="Carbs")
    elif args.meal_type == 'fat':
        axs02.bar(meals_p[:, 0], meals_p[:, 2], color=f_colors[1], width=0.05, label="Fat")
    axs02.set_ylabel(ylabel="Stacked meals (g)")
    axs02.set_ylim(0, 50)
    axs02.legend(loc='upper right')

    if args.meal_type == 'carbs':
        plot_gp_pred(axs[1], x_p, f_means_p[1], f_vars_p[1], color=f_colors[1], label=f_labels[1], lw=2, facecolor=f_colors[1], alpha=0.2, plot_var=plot_var)
        plot_gp_pred(axs[1], x_p, f_means_p_full[1], f_vars_p_full[1], color=f_colors_full[1], label=f_labels_full[1], linestyle='dashed', hatch = 'oo', plot_var=plot_var)
        plot_gp_pred(axs[1], x_p, f_means_p_full[2], f_vars_p_full[2], color=f_colors_full[2], label=f_labels_full[2], lw=5.0,
                     linestyle='dashed', hatch='oo', zorder=10, plot_var=False)
    elif args.meal_type == 'fat':
        plot_gp_pred(axs[1], x_p, f_means_p[1], f_vars_p[1], color=f_colors[1], label=f_labels[1], lw=3, facecolor=f_colors[1], alpha=0.2, plot_var=plot_var)
        plot_gp_pred(axs[1], x_p, f_means_p_full[2], f_vars_p_full[2], color=f_colors_full[2], label=f_labels_full[2], linestyle='dashed', hatch = 'oo', plot_var=plot_var)
    axs[1].legend(loc='upper left')
    axs[1].set(ylabel="Glucose (mmol/l)", xlabel="Time (hours)")
    axs[1].title.set_text('Glucose response to carbs, when fat added.')
    axs12 = axs[1].twinx()
    # if args.meal_type == 'carbs':
    #     axs12.bar(meals_p[:, 0], meals_p[:, 1], color=f_colors[1], width=0.05, label="Carbs")
    # elif args.meal_type == 'fat':
    #     axs12.bar(meals_p[:, 0], meals_p[:, 2], color=f_colors[1], width=0.05, label="Fat")
    axs12.bar(meals_p[:, 0], meals_p[:, 1], color='indigo', width=0.05, label="Carbs")
    axs12.bar(meals_p[:, 0], meals_p[:, 2], bottom=meals_p[:, 1], color='hotpink', width=0.05, label='Fat')
    axs12.set_ylabel(ylabel="Stacked meals (g)")
    axs12.set_ylim(0,50)
    axs12.legend(loc='upper right')

    path = path + 'id' + str(idx) + '/'
    os.makedirs(path, exist_ok=True)
    plt.savefig(path+'predictions_'+time+'_comparison.jpg')
    plt.close()

def plot_gp_pred(ax1, x_p, f_mean, f_var, color='blue', label='Predicted', lw=2, linestyle='solid', hatch='', facecolor='none', alpha=1.0, zorder=1, plot_var=True):
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
    ax1.plot(x_p, f_mean, color, lw=lw, label=label, linestyle=linestyle, zorder=zorder)
    if plot_var:
        ax1.fill_between(
            x_p[:, 0],
            f_mean[:, 0],
            edgecolor=color,
            facecolor = facecolor,
            alpha=alpha,
            hatch=hatch
        )
from matplotlib import pyplot as plt
import os
import numpy as np
import argparse
import pandas as pd
from src.utils.GPResp.data_preparation import arrays_preparation
from matplotlib.lines import Line2D

parser = argparse.ArgumentParser('Plotting original data.')
parser.add_argument('--processed_data', type=str, default='./data/real/processed_data/',
                    help="Path to take processed data.")


def plot_original_data(df_train, df_test):
    plt.rcParams.update({
        "font.family": "Times New Roman",  # use serif/main font for text elements
        "font.size": "16",
        "text.usetex": True,  # use inline math for ticks
        "pgf.rcfonts": False,  # don't setup fonts from rc parameters
        "axes.titlesize": "16",
        "axes.labelsize": "16"
    })

    x, y, meals, patients, P = arrays_preparation(df_train)
    x_test, y_test, meals_test, _, _ = arrays_preparation(df_test)
    plot_original_data_indiv(y, meals, P, 'Train')
    plot_original_data_indiv(y_test, meals_test, P, 'Test')

def plot_original_data_indiv(y, meals, P, type):
    fig, axs = plt.subplots(3, 12, figsize=(13.0, 7.0))
    fig.suptitle(type+ " data distribution")

    for p in range(P):
        axs[0, p].hist(y[p], color='royalblue', alpha=0.7)
        axs[0, p].set_title("Patient {}".format(p+1), fontsize=14)
        axs[1, p].hist(meals[p][:,1], color='darkmagenta', alpha=0.7)
        axs[2, p].hist(meals[p][:,2], color='orange', alpha=0.7)

        if type == 'Train':
            axs[0, p].set_ylim(0, 90)
            axs[0, p].set_xlim(2.0, 8.0)
            axs[1, p].set_ylim(0, 10)
            axs[1, p].set_xlim(0, 40)
            axs[2, p].set_ylim(0, 10)
            axs[2, p].set_xlim(0, 22)
        else:
            axs[0, p].set_ylim(0, 80)
            axs[0, p].set_xlim(2.0, 8.0)
            axs[1, p].set_ylim(0, 4.0)
            axs[1, p].set_xlim(0, 40)
            axs[2, p].set_ylim(0, 8)
            axs[2, p].set_xlim(0, 22)

        if p != 0:
            axs[0, p].yaxis.set_ticklabels([])
            axs[1, p].yaxis.set_ticklabels([])
            axs[2, p].yaxis.set_ticklabels([])

            axs[0, p].xaxis.set_ticklabels([])
            axs[1, p].xaxis.set_ticklabels([])
            axs[2, p].xaxis.set_ticklabels([])

    legend_elements = [Line2D([0], [0], color='royalblue', lw=4, label='Glucose'),
                       Line2D([0], [0], color='darkmagenta', lw=4, label='Carbs'),
                       Line2D([0], [0], color='orange', lw=4, label='Fat')]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3)

    plt.savefig(args.processed_data + '/summary_statistics_'+type+'_plot.pdf', bbox_inches="tight")
    plt.close()


def summary_statistics(df_train, df_test):
    x, y, meals, patients, P = arrays_preparation(df_train)
    x_test, y_test, meals_test, _, _ = arrays_preparation(df_test)

    data = []
    for p in range(P):
        data.append([len(y[p]), len(meals[p]), np.median(y[p]), round(np.median(meals[p][:,1]),2), round(np.median(meals[p][:,2]),2), len(y_test[p]), len(meals_test[p]), np.median(y_test[p]), round(np.median(meals_test[p][:,1]),2), round(np.median(meals_test[p][:,2]),2)])

    summary_df = pd.DataFrame(data)
    summary_df.columns = ['Glucose # train', 'Treatments # train','Glucose median train','Carbs median train', 'Fat median train', 'Glucose # test', 'Treatments # test','Glucose median test','Carbs median test', 'Fat median test']
    summary_df.to_csv(args.processed_data + '/summary_statistics_data.csv')


if __name__ == "__main__":
    args = parser.parse_args()
    os.chdir("../../")

    # Downloading processed data
    df_train = pd.read_csv(args.processed_data + '/df_sliced.csv')
    df_test = pd.read_csv(args.processed_data + '/df_sliced_test.csv')

    # Create original data summary statistics
    summary_statistics(df_train, df_test)

    # Plot the original data
    plot_original_data(df_train, df_test)

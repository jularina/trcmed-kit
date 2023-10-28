from matplotlib import pyplot as plt
import os
import numpy as np
import argparse
import pandas as pd
from src.utils.GPResp.data_preparation import arrays_preparation
from matplotlib.lines import Line2D
import seaborn as sns

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
    plot_original_data_indiv(y, meals, y_test, meals_test, P)
    #plot_original_data_indiv(y_test, meals_test, P, 'Test')

    glucose, carbs, fat, glucose_test, carbs_test, fat_test = [], [], [], [], [], []
    for p in range(P):
        glucose.append(y[p].flatten())
        glucose_test.append(y_test[p].flatten())
        carbs.append(meals[p][:,1])
        carbs_test.append(meals_test[p][:, 1])
        fat.append(meals[p][:,2])
        fat_test.append(meals_test[p][:, 2])

    glucose = np.concatenate(glucose).ravel()
    glucose_test = np.concatenate(glucose_test).ravel()
    carbs = np.concatenate(carbs).ravel()
    carbs_test = np.concatenate(carbs_test).ravel()
    fat = np.concatenate(fat).ravel()
    fat_test = np.concatenate(fat_test).ravel()

    fig, axs = plt.subplots(1, 3, figsize=(12.0, 3.0))
    axs[0].plot([], [], ' ', label="Glucose:")
    sns.kdeplot(data=glucose, color="#4169E1", alpha=0.3, fill=True, label='Train', lw=0, ax=axs[0])
    sns.kdeplot(data=glucose_test, color='#4169E1', alpha=0.7, fill=True, label='Test', lw=0, ax=axs[0])

    axs[1].plot([], [], ' ', label="Carbs:")
    axs[1].hist(carbs, color='darkmagenta', label='Train', alpha=0.3)
    axs[1].hist(carbs_test, color='darkmagenta', label='Test', alpha=0.8)

    axs[2].plot([], [], ' ', label="Fat:")
    axs[2].hist(fat, color='orange', label='Train', alpha=0.3)
    axs[2].hist(fat_test, color='orange', label='Test', alpha=0.8)
    # axs[2].set_title("Fat", fontsize=14)

    # legend_elements = [Line2D([0], [0], color='royalblue', alpha=0.3, lw=4, label='Train glucose'),
    #                    Line2D([0], [0], color='royalblue', alpha=0.8, lw=4, label='Test glucose'),
    #                    Line2D([0], [0], color='darkmagenta', alpha=0.3, lw=4, label='Train carbs'),
    #                    Line2D([0], [0], color='darkmagenta', alpha=0.8, lw=4, label='Test carbs'),
    #                    Line2D([0], [0], color='orange', lw=4, alpha=0.3, label='Train fat'),
    #                    Line2D([0], [0], color='orange', lw=4, alpha=0.8, label='Test fat')
    #                    ]
    # fig.legend(handles=legend_elements, loc='lower center', ncol=3)
    axs[0].legend(fontsize=12)
    axs[1].legend(fontsize=12)
    axs[2].legend(fontsize=12)

    plt.savefig(args.processed_data + '/summary_statistics_plot.pdf', bbox_inches="tight")
    plt.close()

def plot_original_data_indiv(y, meals, y_test, meals_test, P):
    fig, axs = plt.subplots(3, 12, figsize=(13.0, 8.0))

    for p in range(P):
        # axs[0, p].hist(y[p], color='royalblue', alpha=0.3)
        # axs[0, p].hist(y_test[p], color='royalblue', alpha=0.8)
        sns.kdeplot(data=y[p], color='royalblue', alpha=0.3, fill=True, legend=False, lw=0, ax=axs[0, p])
        sns.kdeplot(data=y_test[p], color='royalblue', alpha=0.7, fill=True, legend=False, lw=0, ax=axs[0, p])
        axs[0, p].set_ylabel('')
        axs[0, p].set_title("Patient {}".format(p+1), fontsize=14)
        axs[1, p].hist(meals[p][:,1], color='darkmagenta', alpha=0.3)
        axs[1, p].hist(meals_test[p][:, 1], color='darkmagenta', alpha=0.8)
        axs[2, p].hist(meals[p][:, 2], color='orange', alpha=0.3)
        axs[2, p].hist(meals_test[p][:,2], color='orange', alpha=0.8)

        # if type == 'Train':
        #     axs[0, p].set_ylim(0, 90)
        #     axs[0, p].set_xlim(2.0, 8.0)
        #     axs[1, p].set_ylim(0, 10)
        #     axs[1, p].set_xlim(0, 40)
        #     axs[2, p].set_ylim(0, 10)
        #     axs[2, p].set_xlim(0, 22)
        # else:
        #     axs[0, p].set_ylim(0, 80)
        #     axs[0, p].set_xlim(2.0, 8.0)
        #     axs[1, p].set_ylim(0, 4.0)
        #     axs[1, p].set_xlim(0, 40)
        #     axs[2, p].set_ylim(0, 8)
        #     axs[2, p].set_xlim(0, 22)

        axs[0, p].set_ylim(0, 1.2)
        axs[0, p].set_xlim(2.0, 10.0)
        axs[1, p].set_ylim(0, 10)
        axs[1, p].set_xlim(0, 40)
        axs[2, p].set_ylim(0, 10)
        axs[2, p].set_xlim(0, 22)

        if p != 0:
            axs[0, p].yaxis.set_ticklabels([])
            axs[1, p].yaxis.set_ticklabels([])
            axs[2, p].yaxis.set_ticklabels([])

            axs[0, p].xaxis.set_ticklabels([])
            axs[1, p].xaxis.set_ticklabels([])
            axs[2, p].xaxis.set_ticklabels([])

    legend_elements = [Line2D([0], [0], color='royalblue', alpha=0.3, lw=4, label='Train glucose'),
                       Line2D([0], [0], color='royalblue', alpha=0.8, lw=4, label='Test glucose'),
                       Line2D([0], [0], color='darkmagenta', alpha=0.3, lw=4, label='Train carbs'),
                       Line2D([0], [0], color='darkmagenta', alpha=0.8, lw=4, label='Test carbs'),
                       Line2D([0], [0], color='orange', lw=4, alpha=0.3, label='Train fat'),
                       Line2D([0], [0], color='orange', lw=4, alpha=0.8, label='Test fat')
                       ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3)

    plt.savefig(args.processed_data + '/summary_statistics_plot_indiv.pdf', bbox_inches="tight")
    plt.close()


def summary_statistics(df_train, df_test):
    x, y, meals, patients, P = arrays_preparation(df_train)
    x_test, y_test, meals_test, _, _ = arrays_preparation(df_test)

    data = []
    for p in range(P):
        data.append([len(y[p]), len(meals[p]), np.median(y[p]), round(np.median(meals[p][:,1]),2), round(np.median(meals[p][:,2]),2), len(y_test[p]), len(meals_test[p]), np.median(y_test[p]), round(np.median(meals_test[p][:,1]),2), round(np.median(meals_test[p][:,2]),2)])

    summary_df = pd.DataFrame(data)
    summary_df.columns = ['glucnumtrain', 'treatnumtrain','glucmedtrain','carbsmedtrain', 'fatmedtrain', 'glucnumtest', 'treatnumtest','glucmedtest','carbsmedtest', 'fatmedtest']
    summary_df.to_csv(args.processed_data + '/summary_statistics_data.csv', index=False)


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

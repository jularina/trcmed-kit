from matplotlib import pyplot as plt
import os
import numpy as np


def figure_4a(times, obs, meals, gpconv, gplfm, path):
    plt.rcParams.update({
        "font.family": "Times New Roman",  # use serif/main font for text elements
        "font.size": "16",
        "text.usetex": True,  # use inline math for ticks
        "pgf.rcfonts": False,  # don't setup fonts from rc parameters
        "axes.titlesize": "16",
        "axes.labelsize": "16"
    })
    gpconv_baseline, gpconv_wfat, gpconv_wofat = gpconv['baseline'], gpconv['fitted_glucose'], gpconv[
        'fitted_glucose_wofat']
    gplfm_baseline, gplfm_carbs, gplfm_fat, gplfm_glucose = gplfm['baseline'], gplfm['carbs'], gplfm['fat'], gplfm[
        'fitted_glucose']
    gpconv_baseline_var, gpconv_wfat_var, gpconv_wofat_var = gpconv['baseline_var'], gpconv['fitted_glucose_var'], gpconv[
        'fitted_glucose_wofat_var']
    gplfm_baseline_var, gplfm_carbs_var, gplfm_fat_var, gplfm_glucose_var = gplfm['baseline_var'], gplfm['carbs_var'], gplfm['fat_var'], gplfm[
        'fitted_glucose_var']

    f, (a0, a1, a2) = plt.subplots(3, 1, figsize=(9.0, 4.5), height_ratios=[2, 3, 1], dpi=300, sharex=True)
    plt.xlim(48.0, times.max())

    a0.plot(times, obs, 'x', ms=7, alpha=0.65, label='true observations', c='grey')
    a0.plot(times, gpconv_baseline, color='grey', lw=2.0, label='baseline', linestyle='solid')
    a0.fill_between(times[:, 0],
        gpconv_baseline[:, 0] - 1.96 * np.sqrt(gpconv_baseline_var[:, 0]),
        gpconv_baseline[:, 0] + 1.96 * np.sqrt(gpconv_baseline_var[:, 0]),
        color='grey',
        alpha=0.2,
    )
    a0.plot(times, gpconv_wofat, color='dodgerblue', lw=2.0, label='fitted glucose without fat', linestyle='solid')
    a0.fill_between(times[:, 0],
        gpconv_baseline[:, 0],
        gpconv_wofat[:, 0],
        color='dodgerblue',
        alpha=0.3,
    )
    a0.plot(times, gpconv_wfat, color='royalblue', lw=2.0, label='fitted glucose with fat', linestyle='solid')
    a0.fill_between(times[:, 0],
        gpconv_wfat[:, 0] - 1.96 * np.sqrt(gpconv_wfat_var[:, 0]),
        gpconv_wfat[:, 0] + 1.96 * np.sqrt(gpconv_wfat_var[:, 0]),
        color='royalblue',
        alpha=0.2,
    )
    a0.legend(loc='upper right', fontsize=8, frameon=False)
    a0.set(ylabel="Glucose (mmol/l)")
    a0.set_ylim(3, 6)
    a0.set_title('GP-Conv glucose response to meals with and without fat')

    a1.plot(times, obs, 'x', ms=7, alpha=0.65, label='true observations', c='grey')
    a1.plot(times, gplfm_baseline, color='grey', lw=2.0, label='baseline', linestyle='solid')
    a1.fill_between(times[:, 0],
        gplfm_baseline[:, 0] - 1.96 * np.sqrt(gplfm_baseline_var[:, 0]),
        gplfm_baseline[:, 0] + 1.96 * np.sqrt(gplfm_baseline_var[:, 0]),
        color='grey',
        alpha=0.2,
    )
    a1.plot(times, gplfm_carbs, color='darkmagenta', lw=2.0, label='response to carbs', linestyle='solid')
    a1.fill_between(times[:, 0],
        gplfm_carbs[:, 0] - 1.96 * np.sqrt(gplfm_carbs_var[:, 0]),
        gplfm_carbs[:, 0] + 1.96 * np.sqrt(gplfm_carbs_var[:, 0]),
        color='darkmagenta',
        alpha=0.2,
    )
    a1.plot(times, gplfm_fat, color='orange', lw=2.0, label='response to fat', linestyle='solid')
    a1.fill_between(times[:, 0],
        gplfm_fat[:, 0] - 1.96 * np.sqrt(gplfm_fat_var[:, 0]),
        gplfm_fat[:, 0] + 1.96 * np.sqrt(gplfm_fat_var[:, 0]),
        color='orange',
        alpha=0.2,
    )
    a1.plot(times, gplfm_glucose, color='royalblue', lw=2.0, label='fitted glucose', linestyle='solid')
    a1.fill_between(times[:, 0],
        gplfm_glucose[:, 0] - 1.96 * np.sqrt(gplfm_glucose_var[:, 0]),
        gplfm_glucose[:, 0] + 1.96 * np.sqrt(gplfm_glucose_var[:, 0]),
        color='royalblue',
        alpha=0.2,
    )
    a1.legend(loc='upper right', fontsize=8, frameon=False)
    a1.set(ylabel="Glucose (mmol/l)")
    a1.set_title('GP-LFM glucose response to carbs and fat')

    a2.bar(meals[:, 0], meals[:, 1], color='darkmagenta', width=0.3, label="Carbs")
    a2.bar(meals[:, 0], meals[:, 2], bottom=meals[:, 1], color='orange', width=0.3, label='Fat')
    a2.set(xlabel="Time (hours)", ylabel="Stacked \n meals (g)")
    a2.grid(which='major', color='#DDDDDD', linewidth=0.8)
    a2.legend(loc='upper right', fontsize=10)

    os.makedirs(path, exist_ok=True)
    plt.savefig(path + 'figure_4a.pdf')
    plt.close()


def figure_4b(times, meals, gpconv, gplfm):
    pass


def figure_7(times, obs, meals, models):
    pass


if __name__ == "__main__":
    os.chdir("../../")

    # Plot Figure 4a
    path_original = './data/real/processed_data/patients_arrays/12.npz'
    path_gplfm = './data/real/results_data/non_parametric/GPLFM/patients_arrays/12_test_full.npz'
    path_gpconv = './data/real/results_data/non_parametric/GPConv/patients_arrays/12_test_full.npz'
    path_save = './data/real/results_data/non_parametric/paper_figures/'

    gplfm_data = np.load(path_gplfm)
    gpconv_data = np.load(path_gpconv)
    data_original = np.load(path_original)

    times, obs, meals = data_original['x_test'], data_original['y_test'], data_original['meals_test'],

    figure_4a(times, obs, meals, gpconv_data, gplfm_data, path_save)

    # Plot Figure 4b

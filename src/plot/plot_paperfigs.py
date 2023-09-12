from matplotlib import pyplot as plt
import os
import numpy as np


def despine(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

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

    f, (a0, a1, a2) = plt.subplots(3, 1, figsize=(8.0, 5.0), height_ratios=[2, 4.5, 1], dpi=300, sharex=True)
    plt.xlim(48.0, times.max())
    f.text(0.02, 0.6, 'Glucose (mmol/l)', va='center', rotation='vertical')
    f.tight_layout()

    def plot_trueobs(ax):
        ax.plot(times, obs, 'x', ms=7, alpha=0.65, label='true observations', c='grey')

    plot_trueobs(a0)
    a0.plot(times, gpconv_wfat, color='royalblue', lw=2.0, label='fit with fat', linestyle='solid', zorder=4)
    a0.fill_between(times[:, 0],
        gpconv_wfat[:, 0] - 1.96 * np.sqrt(gpconv_wfat_var[:, 0]),
        gpconv_wfat[:, 0] + 1.96 * np.sqrt(gpconv_wfat_var[:, 0]),
        color='royalblue',
        alpha=0.2,
    )
    a0.plot(times, gpconv_wofat, color='dodgerblue', lw=2.0, label='fit without fat', linestyle='solid', zorder=3)
    a0.fill_between(times[:, 0],
        gpconv_baseline[:, 0],
        gpconv_wofat[:, 0],
        color='dodgerblue',
        alpha=0.3,
    )
    a0.plot(times, gpconv_baseline, color='grey', lw=2.0, label='baseline', linestyle='solid')
    a0.fill_between(times[:, 0],
        gpconv_baseline[:, 0] - 1.96 * np.sqrt(gpconv_baseline_var[:, 0]),
        gpconv_baseline[:, 0] + 1.96 * np.sqrt(gpconv_baseline_var[:, 0]),
        color='grey',
        alpha=0.2,
    )
    a0.legend(loc='right', fontsize=10, framealpha=0.9)
    #a0.set(ylabel="Glucose (mmol/l)")
    a0.set_ylim(3, 6)
    a0.set_title(r'$\texttt{GP-Conv}$ fitted glucose response to meals with and without fat', loc='left')
    despine(a0)

    plot_trueobs(a1)
    a1.plot(times, gplfm_glucose, color='royalblue', lw=2.0, label='fit (total)', linestyle='solid', zorder=4)
    a1.fill_between(times[:, 0],
        gplfm_glucose[:, 0] - 1.96 * np.sqrt(gplfm_glucose_var[:, 0]),
        gplfm_glucose[:, 0] + 1.96 * np.sqrt(gplfm_glucose_var[:, 0]),
        color='royalblue',
        alpha=0.2,
    )
    a1.plot(times, gplfm_baseline, color='grey', lw=2.0, label='baseline', linestyle='solid')
    a1.fill_between(times[:, 0],
        gplfm_baseline[:, 0] - 1.96 * np.sqrt(gplfm_baseline_var[:, 0]),
        gplfm_baseline[:, 0] + 1.96 * np.sqrt(gplfm_baseline_var[:, 0]),
        color='grey',
        alpha=0.2,
    )
    a1.plot(times, gplfm_fat, color='orange', lw=2.0, label='response to fat', linestyle='solid', zorder=3)
    a1.fill_between(times[:, 0],
        gplfm_fat[:, 0] - 1.96 * np.sqrt(gplfm_fat_var[:, 0]),
        gplfm_fat[:, 0] + 1.96 * np.sqrt(gplfm_fat_var[:, 0]),
        color='orange',
        alpha=0.2,
    )
    a1.plot(times, gplfm_carbs, color='darkmagenta', lw=2.0, label='response to carbs', linestyle='solid')
    a1.fill_between(times[:, 0],
        gplfm_carbs[:, 0] - 1.96 * np.sqrt(gplfm_carbs_var[:, 0]),
        gplfm_carbs[:, 0] + 1.96 * np.sqrt(gplfm_carbs_var[:, 0]),
        color='darkmagenta',
        alpha=0.2,
    )
    a1.legend(loc='right', fontsize=10, framealpha=0.9)
    a1.set_yticks([0, 4, 6])
    #a1.set(ylabel="Glucose (mmol/l)")
    a1.set_title(r'$\texttt{GP-LFM}$ fitted glucose response to carbohydrates and fat', loc='left')
    despine(a1)

    a2.bar(meals[:, 0], meals[:, 2], bottom=meals[:, 1], color='orange', width=0.3, label='Fat')
    a2.bar(meals[:, 0], meals[:, 1], color='darkmagenta', width=0.3, label="Carbohydrates")
    a2.set(xlabel="Time (hours)")#, ylabel="Meals (g)")
    a2.set_title("Meal intake", loc='left')
    #a2.grid(which='major', color='#DDDDDD', linewidth=0.8)
    a2.legend(loc='right', fontsize=8, frameon=False)
    a2.set_yticks([0, 25, 50], ["0", r"$25\,$g", r"$50\,$g"])
    despine(a2)


    os.makedirs(path, exist_ok=True)
    plt.savefig(path + 'figure_4a.pdf', bbox_inches="tight")
    plt.close()


def figure_4b(times, meals, gpconv, gplfm, path):
    plt.rcParams.update({
        "font.family": "Times New Roman",  # use serif/main font for text elements
        "font.size": "16",
        "legend.title_fontsize": "10",
        "text.usetex": True,  # use inline math for ticks
        "pgf.rcfonts": False,  # don't setup fonts from rc parameters
        "axes.titlesize": "16",
        "axes.labelsize": "16"
    })
    gpconv_wfat, gpconv_wofat = gpconv['glucose'], gpconv[
        'glucose_wofat']
    gplfm_carbs, gplfm_fat = gplfm['carbs'], gplfm['fat']
    gpconv_wfat_var, gpconv_wofat_var = gpconv['glucose_var'], gpconv[
        'glucose_wofat_var']
    gplfm_carbs_var, gplfm_fat_var = gplfm['carbs_var'], gplfm['fat_var']

    f, (a0, a1) = plt.subplots(2, 1, figsize=(4.0, 5.0), dpi=300, sharex=True)
    f.text(0.0, 0.5, 'Meal intake (g)', va='center', rotation='vertical')
    f.text(1.0, 0.5, 'Glucose response (mmol/l)', va='center', rotation='vertical')
    f.tight_layout()

    def plot_meal_bars(ax):
        ax.bar(meals[:, 0], meals[:, 1], color='darkmagenta', width=0.3, label="Carbs")
        ax.bar(meals[:, 0], meals[:, 2], bottom=meals[:, 1], color='orange', width=0.3, label='Fat')
        ax.text(meals[:, 0], meals[:, 1]/2,             f"carbs.\n{meals[0,1]:.1f}", ha='center', va='center', rotation='vertical', fontsize=8, color='white')
        ax.text(meals[:, 0], meals[:, 1]+meals[:, 2]/2, f"fat\n{meals[0,2]:.1f}", ha='center', va='center', rotation='vertical', fontsize=8)

    plot_meal_bars(a0)
    a0.set_xlim(-0.3, 3)
    a0.set_ylim(0,50)
    a0.set_title(r'$\texttt{GP-Conv}$')

    a01 = a0.twinx()
    a01.plot(times, gpconv_wfat, color='royalblue', lw=2.0, label='with fat', linestyle='solid')
    a01.fill_between(times[:, 0],
        gpconv_wfat[:, 0] - 1.96 * np.sqrt(gpconv_wfat_var[:, 0]),
        gpconv_wfat[:, 0] + 1.96 * np.sqrt(gpconv_wfat_var[:, 0]),
        color='royalblue',
        alpha=0.2,
    )
    a01.plot(times, gpconv_wofat, color='lightblue', lw=2.0, label='without fat', linestyle='solid')
    a01.fill_between(times[:, 0],
        gpconv_wofat[:, 0] - 1.96 * np.sqrt(gpconv_wofat_var[:, 0]),
        gpconv_wofat[:, 0] + 1.96 * np.sqrt(gpconv_wofat_var[:, 0]),
        color='dodgerblue',
        alpha=0.2,
    )
    a01.set(xlabel="Time (hours)")
    a01.legend(loc='right', fontsize=10, frameon=False, title="fitted response")

    a0.set_zorder(a01.get_zorder()+1)  # have bars (a0) on top of lines (a01)
    a0.patch.set_visible(False)  # need to remove white background of a0

    plot_meal_bars(a1)
    a1.set(xlabel="Time (hours)")
    a1.set_ylim(0,50)
    a1.set_title(r'$\texttt{GP-LFM}$')
    a11 = a1.twinx()
    a11.plot(times, gplfm_carbs, color='darkmagenta', lw=2.0, label='carbohydrates', linestyle='solid')
    a11.fill_between(times[:, 0],
        gplfm_carbs[:, 0] - 1.96 * np.sqrt(gplfm_carbs_var[:, 0]),
        gplfm_carbs[:, 0] + 1.96 * np.sqrt(gplfm_carbs_var[:, 0]),
        color='darkmagenta',
        alpha=0.2,
    )
    a11.plot(times, gplfm_fat, color='orange', lw=2.0, label='fat', linestyle='solid')
    a11.fill_between(times[:, 0],
        gplfm_fat[:, 0] - 1.96 * np.sqrt(gplfm_fat_var[:, 0]),
        gplfm_fat[:, 0] + 1.96 * np.sqrt(gplfm_fat_var[:, 0]),
        color='orange',
        alpha=0.2,
    )
    a11.set(xlabel="Time (hours)")
    a11.legend(loc='right', fontsize=10, frameon=False, title="fitted response to")
    
    a1.set_zorder(a11.get_zorder()+1)  # have bars (a1) on top of lines (a11)
    a1.patch.set_visible(False)  # need to remove white background of a1

    os.makedirs(path, exist_ok=True)
    plt.savefig(path + 'figure_4b.pdf', bbox_inches="tight")
    plt.close()

def figure_7(times, obs, meals, cheng, hizli, gpresp, gplfm, gpconv, path):
    plt.rcParams.update({
        "font.family": "Times New Roman",  # use serif/main font for text elements
        "font.size": "16",
        "text.usetex": True,  # use inline math for ticks
        "pgf.rcfonts": False,  # don't setup fonts from rc parameters
        "axes.titlesize": "16",
        "axes.labelsize": "16"
    })
    cheng_glucose = cheng['fitted_glucose']
    hizli_glucose = hizli['fitted_glucose']
    gpresp_glucose = gpresp['fitted_glucose']
    gpconv_glucose = gpconv['fitted_glucose']
    gplfm_glucose = gplfm['fitted_glucose']


    f, (a0, a1) = plt.subplots(2, 1, figsize=(8.0, 4.0), height_ratios=[3, 1], dpi=300, sharex=True)
    plt.xlim(48.0, times.max())
    f.tight_layout()

    a0.plot(times, obs, 'x', ms=7, alpha=0.65, label='true observations', c='grey')
    a0.plot(times, cheng_glucose, color='yellow', lw=2.0, label='Cheng et al.', linestyle='solid')
    a0.plot(times, hizli_glucose, color='red', lw=2.0, label='Hizli et al.', linestyle='solid')
    a0.plot(times, gpresp_glucose, color='green', lw=2.0, label='GP-Resp', linestyle='solid')
    a0.plot(times, gplfm_glucose, color='lightblue', lw=2.0, label='GP-LFM', linestyle='solid')
    a0.plot(times, gpconv_glucose, color='royalblue', lw=2.0, label='GP-Conv', linestyle='solid')
    a0.legend(loc='upper right', fontsize=10)
    a0.set(ylabel="Glucose (mmol/l)")


    a1.bar(meals[:, 0], meals[:, 1], color='darkmagenta', width=0.3, label="Carbs")
    a1.bar(meals[:, 0], meals[:, 2], bottom=meals[:, 1], color='orange', width=0.3, label='Fat')
    a1.set(xlabel="Time (hours)", ylabel="Stacked \n meals (g)")
    a1.grid(which='major', color='#DDDDDD', linewidth=0.8)
    a1.legend(loc='upper right', fontsize=8)

    os.makedirs(path, exist_ok=True)
    plt.savefig(path + 'figure_7_p57.pdf', bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    os.chdir("../../")

    # Plot Figure 4a
    path_original = './data/real/processed_data/patients_arrays/29.npz'
    path_gplfm = './data/real/results_data/non_parametric/GPLFM/patients_arrays/29_test_full.npz'
    path_gpconv = './data/real/results_data/non_parametric/GPConv/patients_arrays/29_test_full.npz'
    path_save = './data/real/results_data/non_parametric/paper_figures/'

    gplfm_data = np.load(path_gplfm)
    gpconv_data = np.load(path_gpconv)
    data_original = np.load(path_original)

    times, obs, meals = data_original['x_test'], data_original['y_test'], data_original['meals_test']

    figure_4a(times, obs, meals, gpconv_data, gplfm_data, path_save)

    # Plot Figure 4b
    path_original = './data/real/processed_data/patients_arrays/29_onemeal.npz'
    path_gplfm = './data/real/results_data/non_parametric/GPLFM/patients_arrays/29_test_onemeal.npz'
    path_gpconv = './data/real/results_data/non_parametric/GPConv/patients_arrays/29_test_onemeal.npz'
    path_save = './data/real/results_data/non_parametric/paper_figures/'

    gplfm_data = np.load(path_gplfm)
    gpconv_data = np.load(path_gpconv)
    data_original = np.load(path_original)

    times, meals = data_original['x_test_meal'], data_original['meals_test_meal']

    figure_4b(times, meals, gpconv_data, gplfm_data, path_save)

    # Plot Figure 7
    path_original = './data/real/processed_data/patients_arrays/57.npz'
    path_cheng = './data/real/results_data/non_parametric/Chengetal/patients_arrays/57_test.npz'
    path_hizli = './data/real/results_data/non_parametric/Hizlietal/patients_arrays/57_test.npz'
    path_gpresp = './data/real/results_data/non_parametric/GPResp/patients_arrays/57_test.npz'
    path_gplfm = './data/real/results_data/non_parametric/GPLFM/patients_arrays/57_test_full.npz'
    path_gpconv = './data/real/results_data/non_parametric/GPConv/patients_arrays/57_test_full.npz'
    path_save = './data/real/results_data/non_parametric/paper_figures/'

    cheng_data = np.load(path_cheng)
    hizli_data = np.load(path_hizli)
    gpresp_data = np.load(path_gpresp)
    gplfm_data = np.load(path_gplfm)
    gpconv_data = np.load(path_gpconv)
    data_original = np.load(path_original)

    times, obs, meals = data_original['x_test'], data_original['y_test'], data_original['meals_test']

    figure_7(times, obs, meals, cheng_data, hizli_data, gpresp_data, gplfm_data, gpconv_data, path_save)

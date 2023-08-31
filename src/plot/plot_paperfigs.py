from matplotlib import pyplot as plt
import os


def figure_4a(times, obs, meals, gpconv, gplfm, path):
    plt.rcParams.update({
        "font.family": "Times New Roman",  # use serif/main font for text elements
        "font.size": "16",
        "text.usetex": True,  # use inline math for ticks
        "pgf.rcfonts": False,  # don't setup fonts from rc parameters
        "axes.titlesize": "16",
        "axes.labelsize": "16"
    })
    gpconv_baseline, gpconv_wfat, gpconv_wofat = gpconv[0], gpconv[1], gpconv[2]
    gplfm_baseline, gplfm_carbs, gplfm_fat, gplfm_glucose = gplfm[0], gplfm[1], gplfm[2], gplfm[3]

    f, (a0, a1, a2) = plt.subplots(3, 1, figsize=(8.0, 4.0), height_ratios=[2, 3, 1], dpi=300, sharex=True)
    plt.xlim(48.0, times.max())

    a0.plot(times, obs, 'x', ms=7, alpha=0.65, label='true observations', c='grey')
    a0.plot(times, gpconv_baseline, color='grey', lw=2.0, label='baseline', linestyle='solid')
    a0.plot(times, gpconv_wofat, color='dodgerblue', lw=2.0, label='fitted glucose without fat', linestyle='solid')
    a0.plot(times, gpconv_wfat, color='royalblue', lw=2.0, label='fitted glucose with fat', linestyle='solid')
    a0.legend(loc='upper right', fontsize=8, frameon=False)
    a0.set(ylabel="Glucose (mmol/l)")
    a0.set_ylim(3, 6)
    a0.set_title('GP-Conv glucose response to meals with and without fat')

    a1.plot(times, obs, 'x', ms=7, alpha=0.65, label='true observations', c='grey')
    a1.plot(times, gplfm_baseline, color='grey', lw=2.0, label='baseline', linestyle='solid')
    a1.plot(times, gplfm_carbs, color='darkmagenta', lw=2.0, label='response to carbs', linestyle='solid')
    a1.plot(times, gplfm_fat, color='orange', lw=2.0, label='response to fat', linestyle='solid')
    a1.plot(times, gplfm_glucose, color='royalblue', lw=2.0, label='fitted glucose', linestyle='solid')
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

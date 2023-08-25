import os
from matplotlib import pyplot as plt
import tensorflow as tf
import matplotlib.ticker as tick

def plot_treatment_kernel(vals, mask, meal, colors):
    masked_vals = tf.boolean_mask(vals, mask)
    mat = tf.linalg.tensor_diag(masked_vals)
    plt.figure(figsize=(15, 15))
    plt.matshow(mat, cmap=colors)
    plt.title(r'$K_{r}$' + ' for test ' + meal + ' (diag.)',
              weight='bold', fontsize=13)
    plt.colorbar()
    path = './data/results_data/non_parametric/GPLFM/kernels/'
    os.makedirs(path, exist_ok=True)
    plt.savefig(path+'Kr_test_'+meal+'.png', bbox_inches='tight')
    plt.close()

def plot_baseline_kernel_train(vals, mask):
    mat = tf.boolean_mask(vals, mask)
    mat = tf.boolean_mask(mat, mask, axis=1)
    plt.figure(figsize=(15, 15))
    plt.matshow(mat, cmap='Purples')
    plt.title(r'$K_{b}$' + ' for train',
              weight='bold', fontsize=13)
    plt.colorbar()
    path = './data/results_data/non_parametric/GPLFM/kernels/'
    os.makedirs(path, exist_ok=True)
    plt.savefig(path+'Kb_train.png', bbox_inches='tight')
    plt.close()

def plot_total_kernel(vals, mask):
    mat = tf.boolean_mask(vals, mask)
    mat = tf.boolean_mask(mat, mask, axis=1)
    plt.figure(figsize=(15, 15))
    plt.matshow(mat, cmap='Purples')
    plt.title(r'$K_{t}$' + ' for train',
              weight='bold', fontsize=13)
    plt.colorbar()
    path = '../../../data/results_data/non_parametric/GPLFM/kernels/'
    os.makedirs(path, exist_ok=True)
    plt.savefig(path+'Kt_train.png', bbox_inches='tight')
    plt.close()

def plot_treatment_kernel_train(vals, mask):
    mat = tf.boolean_mask(vals, mask)
    mat = tf.boolean_mask(mat, mask, axis=1)
    plt.figure(figsize=(15, 15))
    plt.matshow(mat, cmap='Purples')
    plt.title(r'$K_{r}$' + ' for train',
              weight='bold', fontsize=13)
    plt.colorbar()
    path = '../../../data/results_data/non_parametric/GPLFM/kernels/'
    os.makedirs(path, exist_ok=True)
    plt.savefig(path+'Kr_train.png', bbox_inches='tight')
    plt.close()

def plot_baseline_kernel(mat):
    mat = tf.linalg.tensor_diag(mat)
    plt.figure(figsize=(15, 15))
    plt.matshow(mat, cmap='Purples')
    plt.title(r'$K_{b}$' + ' for test (diag.)',
              weight='bold', fontsize=13)
    plt.colorbar()
    path = '../../../data/results_data/non_parametric/GPLFM/kernels/'
    os.makedirs(path, exist_ok=True)
    plt.savefig(path+'Kb_test.png', bbox_inches='tight')
    plt.close()

def plot_tlse_kernel(vals, mask, meal, colors):
    mat = tf.boolean_mask(vals, mask)
    mat = tf.boolean_mask(mat, mask, axis=1)
    plt.figure(figsize=(15, 15))
    plt.matshow(mat, cmap=colors)
    plt.title(r'$K_{TLSE}$' + ' for train ' + meal,
              weight='bold', fontsize=13)
    plt.colorbar()
    path = '../../../data/results_data/non_parametric/GPLFM/kernels/'
    os.makedirs(path, exist_ok=True)
    plt.savefig(path+'Ktlse_'+meal+'_train.png', bbox_inches='tight')
    plt.close()

def plot_B_kernel(vals, mask, meal, colors):
    mat = tf.boolean_mask(vals, mask)
    mat = tf.boolean_mask(mat, mask, axis=1)
    plt.figure(figsize=(15, 15))
    plt.matshow(mat, cmap=colors)
    plt.title(r'$B_{stretched}$' + ' for train ' + meal,
              weight='bold', fontsize=13)
    plt.colorbar()
    path = '../../../data/results_data/non_parametric/GPLFM/kernels/'
    os.makedirs(path, exist_ok=True)
    plt.savefig(path+'Bstretched_'+meal+'_train.png', bbox_inches='tight')
    plt.close()

def plot_total_kernel_whole_train(b1, k1, b2, k2, kr, b, kt, mask1, mask):
    fig = plt.figure(figsize=(12, 5))
    ax1 = plt.subplot2grid((5, 5), (0, 0), rowspan=2, colspan=1)
    ax2 = plt.subplot2grid((5, 5), (0, 1), rowspan=2, colspan=1)
    ax3 = plt.subplot2grid((5, 5), (0, 2), rowspan=2, colspan=1)
    ax4 = plt.subplot2grid((5, 5), (0, 3), rowspan=2, colspan=1)
    ax5 = plt.subplot2grid((5, 5), (0, 4), rowspan=2, colspan=1)
    ax6 = plt.subplot2grid((5, 5), (2, 0), rowspan=3, colspan=2)
    ax7 = plt.subplot2grid((5, 5), (2, 2), rowspan=2, colspan=1)
    ax8 = plt.subplot2grid((5, 5), (2, 3), rowspan=3, colspan=2)

    b1 = tf.boolean_mask(b1, mask1)
    b1 = tf.boolean_mask(b1, mask1, axis=1)

    b2 = tf.boolean_mask(b2, mask1)
    b2 = tf.boolean_mask(b2, mask1, axis=1)

    k1 = tf.boolean_mask(k1, mask1)
    k1 = tf.boolean_mask(k1, mask1, axis=1)

    k2 = tf.boolean_mask(k2, mask1)
    k2 = tf.boolean_mask(k2, mask1, axis=1)

    kr = tf.boolean_mask(kr, mask1)
    kr = tf.boolean_mask(kr, mask1, axis=1)

    b = tf.boolean_mask(b, mask)
    b = tf.boolean_mask(b, mask, axis=1)

    kt = tf.boolean_mask(kt, mask)
    kt = tf.boolean_mask(kt, mask, axis=1)

    im1 = ax1.matshow(b1, cmap='PuBu')
    fig.colorbar(im1, ax=ax1)
    ax1.locator_params(nbins=5)
    ax1.set_title(r'$B_{stretched}$ ' + ' carbs',fontsize=13)
    ax1.xaxis.set_ticks_position('bottom')

    im2 = ax2.matshow(k1, cmap='PuBu')
    fig.colorbar(im2, ax=ax2)
    ax2.locator_params(nbins=5)
    ax2.set_title(r'$K_{TLSE}$ ' + ' carbs',fontsize=13)
    ax2.xaxis.set_ticks_position('bottom')

    im3 = ax3.matshow(b2, cmap='RdPu')
    fig.colorbar(im3, ax=ax3)
    ax3.locator_params(nbins=5)
    ax3.set_title(r'$B_{stretched}$ ' + ' fat', fontsize=13)
    ax3.xaxis.set_ticks_position('bottom')

    im4 = ax4.matshow(k2, cmap='RdPu')
    cbar=fig.colorbar(im4, ax=ax4)
    ax4.locator_params(nbins=5)
    cbar.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    ax4.set_title(r'$K_{TLSE}$ ' + ' fat', fontsize=13, pad=0.2)
    ax4.xaxis.set_ticks_position('bottom')

    im5 = ax5.matshow(kr, cmap='Purples')
    fig.colorbar(im5, ax=ax5)
    ax5.locator_params(nbins=5)
    ax5.set_title(r'$K_{r}$', fontsize=13)
    ax5.xaxis.set_ticks_position('bottom')

    im1 = ax6.matshow(b, cmap='Purples')
    cbar = fig.colorbar(im1, ax=ax6)
    ax6.locator_params(nbins=5)
    cbar.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.3f'))
    ax6.ticklabel_format(style='plain')
    ax6.set_title(r'$K_{b}$',fontsize=13)
    ax6.xaxis.set_ticks_position('bottom')

    im2 = ax7.matshow(kr, cmap='Purples')
    fig.colorbar(im2, ax=ax7)
    ax7.locator_params(nbins=5)
    ax7.set_title(r'$K_{r}$',fontsize=13)
    ax7.xaxis.set_ticks_position('bottom')

    im3 = ax8.matshow(kt, cmap='Purples')
    fig.colorbar(im3, ax=ax8)
    ax8.locator_params(nbins=5)
    ax8.set_title(r'$K_{t}$', fontsize=13)
    ax8.xaxis.set_ticks_position('bottom')

    plt.scatter(1.43, 0.55, s=150, c='black', transform=ax1.transAxes, marker='x', clip_on=False)
    plt.scatter(3.2, 0.55, s=150, c='black', transform=ax1.transAxes, marker='+', clip_on=False)
    plt.scatter(4.95, 0.55, s=150, c='black', transform=ax1.transAxes, marker='x', clip_on=False)
    plt.scatter(6.7, 0.55, s=150, c='black', transform=ax1.transAxes, marker='_', clip_on=False)
    plt.scatter(6.7, 0.59, s=150, c='black', transform=ax1.transAxes, marker='_', clip_on=False)

    plt.scatter(1.4, 0.55, s=200, c='black', transform=ax6.transAxes, marker='+', clip_on=False)
    plt.scatter(2.6, 0.55, s=200, c='black', transform=ax6.transAxes, marker='_', clip_on=False)
    plt.scatter(2.6, 0.59, s=200, c='black', transform=ax6.transAxes, marker='_', clip_on=False)

    path = './data/results_data/non_parametric/GPLFM/kernels/'
    os.makedirs(path, exist_ok=True)
    plt.savefig(path+'lfm_model_cov_train.pdf', bbox_inches='tight')
    plt.close()

def plot_total_kernel_whole_test(kr1, kr2, b, mask1, mask2):
    plt.locator_params(nbins=4)
    fig, axs = plt.subplots(1,3,figsize=(10,3))
    fig.tight_layout(pad=2.0)

    kr1 = tf.boolean_mask(kr1, mask1)
    kr1 = tf.linalg.tensor_diag(kr1)

    kr2 = tf.boolean_mask(kr2, mask2)
    kr2 = tf.linalg.tensor_diag(kr2)

    b = tf.linalg.tensor_diag(b)

    im1 = axs[0].matshow(kr1, cmap='PuBu')
    fig.colorbar(im1, ax=axs[0])
    axs[0].set_title(r'$K_{r}$'+' carbs',fontsize=13)

    im2 = axs[1].matshow(kr2, cmap='RdPu')
    fig.colorbar(im2, ax=axs[1])
    axs[1].set_title(r'$K_{r}$'+' fat',fontsize=13)

    im3 = axs[2].matshow(b, cmap='Purples')
    fig.colorbar(im3, ax=axs[2])
    axs[2].set_title(r'$K_{b}$', fontsize=13)

    path = './data/results_data/non_parametric/GPLFM/kernels/'
    os.makedirs(path, exist_ok=True)
    plt.savefig(path+'lfm_model_cov_test.pdf', bbox_inches='tight')
    plt.close()
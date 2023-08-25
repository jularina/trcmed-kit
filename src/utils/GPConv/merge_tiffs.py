from matplotlib import pyplot as plt
import tifffile as tiff

im1 = tiff.imread("../../data/results_data/non-parametric/operation/carbs/id29/Fig10.tiff")
im2 = tiff.imread("../../data/results_data/non-parametric/single_meal/operation/carbs/id29/Fig10.tiff")

fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.0), dpi=300)
fig.tight_layout(pad=0.1)
axs[0].imshow(im1)
axs[0].axis('off')
axs[1].imshow(im2)
axs[1].axis('off')
plt.savefig("../../data/results_data/non-parametric/operation/carbs/id29/Fig10merged.png", bbox_inches='tight')
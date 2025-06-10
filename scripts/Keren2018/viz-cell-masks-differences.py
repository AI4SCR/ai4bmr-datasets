# diagnostic plot
# FIXME: what do to with this?
# note: map objects in mask that are not present in data to 2
objs_in_segm = set(mask[segm == 1])
map = {i: 2 for i in objs_in_segm - objs_in_data}
assert objs_in_data - objs_in_segm == set()
map.update({i: 1 for i in objs_in_data.intersection(objs_in_segm)})
z = mask.copy()
z[segm != 1] = 0
vfunc = np.vectorize(lambda x: map.get(x, x))
z = vfunc(z)

fig, axs = plt.subplots(2, 2, dpi=400)
axs = axs.flat

axs[0].imshow(mask > 1, interpolation=None, cmap="grey")
axs[0].set_title("Segmentation from processed data", fontsize=8)

axs[1].imshow(mask_filtered > 0, interpolation=None, cmap="grey")
axs[1].set_title("Segmentation from raw data", fontsize=8)

axs[2].imshow(z, interpolation=None)
axs[2].set_title("Objects missing in `cellData.csv`", fontsize=8)

axs[3].imshow(mask_filtered_v2 > 0, interpolation=None, cmap="grey")
axs[3].set_title(
    "Segmentation from processed\nfiltered with `cellData.csv`",
    fontsize=8,
)

for ax in axs:
    ax.set_axis_off()
fig.tight_layout()
fig.savefig(self.misc / f"{sample_id}.png")
plt.close(fig)
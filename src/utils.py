import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend('agg')


def gen_fig(inputs, generated, targets):
    r, c = 3, 3
    titles = ['Condition', 'Generated', 'Original']
    all_imgs = np.concatenate([inputs, generated, targets])

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(all_imgs[cnt, :, :, 0], cmap='gray')
            axs[i, j].set_title(titles[i], fontdict={'fontsize': 8})
            axs[i, j].axis('off')
            cnt += 1
    return fig


def gen_fig_seg(inputs, generated, targets, fake_segs, targets_seg, target_seg_gt):
    r, c = 3, 4
    titles_even = ['Condition', 'Generated', 'Original']
    titles_odd = ['GT_Seg', 'Generated_Seg', 'Original_Seg']
    all_imgs_even = np.concatenate([inputs, generated, targets])
    all_imgs_odd = np.concatenate(([target_seg_gt, fake_segs, targets_seg]))

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(0, c, 2):
            axs[i, j].imshow(all_imgs_even[cnt, :, :, 0], cmap='gray')
            axs[i, j].set_title(titles_even[i], fontdict={'fontsize': 8})
            axs[i, j].axis('off')

            axs[i, j + 1].imshow(all_imgs_odd[cnt, :, :, 0], cmap='gray')
            axs[i, j + 1].set_title(titles_odd[i], fontdict={'fontsize': 8})
            axs[i, j + 1].axis('off')

            cnt += 1
    return fig


def rotate_degree(mask):
    from scipy import ndimage
    import math
    [COG_x, COG_y] = ndimage.measurements.center_of_mass(
        mask)  # returns tuple: x,y => first vertical second horizontal
    horizontal_sum = np.sum(mask > 0, axis=1)
    top_x = np.min(np.where(horizontal_sum > 0))
    top_y = np.median(np.where(mask[top_x, :] != 0))

    degree = np.arctan((COG_y - top_y) / (COG_x - top_x))
    return -math.degrees(degree)


def fill_and_get_LCC(seg):  # binary fill and get largest connected component
    from scipy.ndimage.morphology import binary_fill_holes
    from skimage.measure import label
    seg[seg >= 0.5] = 1
    seg[seg < 0.5] = 0
    seg = binary_fill_holes(seg)
    seg = np.uint8(seg)
    if np.sum(seg) > 0:
        labels = label(seg)
        largestCC = labels == np.argmax(np.bincount(labels.flat, weights=seg.flat))
        return largestCC
    return -1


def get_LV_lenght(mask, match_apical_length):
    from scipy.ndimage import rotate
    mask = mask.astype('uint8')
    deg = 0
    if match_apical_length:
        deg = rotate_degree(mask)
        mask = rotate(mask, deg)
    horizontal_sum = np.sum(mask > 0, axis=1)
    horizontal_sum[horizontal_sum > 0] = 1
    L_lenght = np.sum(horizontal_sum)

    return mask, deg, L_lenght

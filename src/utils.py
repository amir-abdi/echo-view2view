import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def gen_fig(inputs, generated, targets, return_all_imgs=False, batch_size=3):
    plt.switch_backend('agg')
    r, c = 3, batch_size
    titles = ['Condition', 'Generated', 'Original']
    all_imgs = np.concatenate([inputs, generated, targets])
    imgs = []

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            img_normalized = (all_imgs[cnt, :, :, 0] / all_imgs[cnt, :, :, 0].max()) * 255.0
            axs[i, j].imshow(img_normalized, cmap='gray')
            axs[i, j].set_title(titles[i], fontdict={'fontsize': 8})
            axs[i, j].axis('off')
            cnt += 1

            if return_all_imgs:
                img = Image.fromarray(img_normalized)
                imgs.append({'img': img.convert('L'),
                            'name': titles[i] + str(j)})
    if return_all_imgs:
        return fig, imgs
    return fig


def gen_fig_seg(inputs, generated, targets, fake_segs, targets_seg, target_seg_gt):
    """

    :param inputs:
    :param generated:
    :param targets:
    :param fake_segs:
    :param targets_seg:
    :param target_seg_gt:
    :return:
    """
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
    """

    :param mask:
    :return:
    """
    from scipy import ndimage
    import math
    # returns tuple: x,y => first vertical second horizontal
    mask = np.squeeze(mask)
    [COG_x, COG_y] = ndimage.measurements.center_of_mass(mask)
    horizontal_sum = np.sum(mask > 0, axis=1)
    top_x = np.min(np.where(horizontal_sum > 0))
    top_y = np.median(np.where(mask[top_x, :] != 0))

    degree = np.arctan((COG_y - top_y) / (COG_x - top_x))
    return -math.degrees(degree)


def fill_and_get_LCC(seg):
    """
    binary fill and get largest connected component
    :param seg:
    :return:
    """
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


def get_LV_lenght(mask, rotate_match):
    from scipy.ndimage import rotate
    mask = mask.astype('uint8')
    deg = 0
    if rotate_match:
        deg = rotate_degree(mask)
        mask = rotate(mask, deg)
    horizontal_sum = np.sum(mask > 0, axis=1)
    horizontal_sum[horizontal_sum > 0] = 1
    L_lenght = np.sum(horizontal_sum)

    return L_lenght, mask, deg


def match_image_size(img, size):
    h, w = size
    imh, imw = img.shape

    if h == imh and w == imw:
        return img

    if h < imh and w < imw:
        diffh = imh - h
        diffw = imw - w
        return img[int(diffh / 2):int(imh - diffh / 2), int(diffw / 2):int(imw - diffw / 2)]

    if h > imh and w > imw:
        import cv2
        new_img = cv2.copyMakeBorder(img,
                                     int((h - imh) / 2) + (imh % 2), int((h - imh) / 2),
                                     int((w - imw) / 2) + (imh % 2), int((w - imw) / 2),
                                     cv2.BORDER_REFLECT)
        return new_img



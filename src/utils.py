import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

# plt.switch_backend('agg')


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


def gen_fig_seg(inputs, generated, targets, fake_segs, targets_seg):

    r, c = 3, 4
    titles_even = ['Condition', 'Generated', 'Original']
    titles_odd = ['Original', 'Generated_Seg', 'Original_Seg']
    all_imgs_even = np.concatenate([inputs, generated, targets])
    all_imgs_odd = np.concatenate(([targets, fake_segs, targets_seg]))

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(0,c,2):
            axs[i, j].imshow(all_imgs_even[cnt, :, :, 0], cmap='gray')
            axs[i, j].set_title(titles_even[i], fontdict={'fontsize': 8})
            axs[i, j].axis('off')

            axs[i, j+1].imshow(all_imgs_odd[cnt, :, :, 0], cmap='gray')
            axs[i, j+1].set_title(titles_odd[i], fontdict={'fontsize': 8})
            axs[i, j+1].axis('off')


            cnt += 1
    return fig
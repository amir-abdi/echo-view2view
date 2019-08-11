import numpy as np
import matplotlib
# matplotlib.use('Agg')
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
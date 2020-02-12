import numpy as np


def create_BCELogit_loss_label(label_size, pos_thr, neg_thr, upscale_factor=4):
    # positive and negative threshold is upsampled value
    # label size is corr_map_sz
    assert upscale_factor != 0, "Upscale Factor not zero!"
    pos_thr = pos_thr * upscale_factor
    neg_thr = neg_thr * upscale_factor
    center = (label_size - 1) / 2
    line = np.arange(0, label_size)
    line = line - center
    line = line**2
    line = np.expand_dims(line, axis=0)
    dist_map = line + line.transpose()
    label = np.zeros([label_size, label_size, 2]).astype(np.float32)
    label[:, :, 0] = dist_map <= pos_thr**2
    label[:, :, 1] = (dist_map <= pos_thr**2) | (dist_map > neg_thr**2)
    return label


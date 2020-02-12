import numpy as np
import PIL
from PIL import Image
from math import floor


PIL_FLAGS = {'bilinear':PIL.Image.BILINEAR, 'bicubic':PIL.Image.BICUBIC, 'nearest':PIL.Image.NEAREST}


def resize(image, target_size, interpolate='bicubic'):
    img = Image.fromarray(image.astype('uint8', copy=False), 'RGB')
    resize_img = np.array(img.resize(target_size, PIL_FLAGS[interpolate]))
    return resize_img


def crop_img(img, cy, cx, reg_s):
    assert reg_s % 2 != 0, "have to odd integer"
    pads = dict({'left': 0, 'right': 0, 'up': 0, 'down': 0})
    h, w, _ = img.shape
    context = (reg_s - 1)/2
    xcrop_min = int(floor(cx) - context)
    xcrop_max = int(floor(cx) + context)
    ycrop_min = int(floor(cy) - context)
    ycrop_max = int(floor(cy) + context)

    if xcrop_min < 0:
        pads['left'] = -xcrop_min
        xcrop_min = 0
    if ycrop_min < 0:
        pads['up'] = -ycrop_min
        ycrop_min = 0
    if xcrop_max >= w:
        pads['right'] = xcrop_max - w + 1
        xcrop_max = w - 1
    if ycrop_max >= h:
        pads['down'] = ycrop_max - h + 1
        ycrop_max = h - 1
    cropped_img = img[ycrop_min:(ycrop_max+1), xcrop_min:(xcrop_max+1), :]
    return cropped_img, pads


def resize_and_pad(cropped_img, out_sz, pads, reg_s=None, use_avg=True, resize_func=None):
    assert resize_func is not None, "resize func have to present"
    cr_h, cr_w, _ = cropped_img.shape
    if reg_s:
        assert ((cr_h+pads['up']+pads['down'] == reg_s) and
                (cr_w+pads['left']+pads['right'] == reg_s)), (
            'The informed crop dimensions and pad amounts are not consistent '
            'with the informed region side. Cropped img shape: {}, Pads: {}, '
            'Region size: {}.'
            .format(cropped_img.shape, pads, reg_s))
    rz_ratio = out_sz/(cr_h + pads['up'] + pads['down'])
    rz_cr_h = round(rz_ratio*cr_h)
    rz_cr_w = round(rz_ratio*cr_w)
    pads['up'] = round(rz_ratio * pads['up'])
    pads['down'] = out_sz - (rz_cr_h + pads['up'])
    pads['left'] = round(rz_ratio * pads['left'])
    pads['right'] = out_sz - (rz_cr_w + pads['left'])
    rz_crop = resize_func(cropped_img, (rz_cr_w, rz_cr_h), interpolate='bilinear')
    if use_avg:
        val = np.mean(cropped_img, axis=(0, 1))
    else:
        val = 0
    if not all(p == 0 for p in pads.values()):
        rz_crop_trans = np.transpose(rz_crop, axes=(2, 0, 1))
        out_img = np.asarray([np.pad(img,  pad_width=((pads['up'], pads['down']),
                                                      (pads['left'], pads['right'])),
                                     mode='constant', constant_values=val[idx])
                              for idx, img in enumerate(rz_crop_trans)])
        out_img = np.transpose(out_img, axes=(1,2,0))
    else:
        out_img = rz_crop

    return out_img
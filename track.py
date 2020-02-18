import os, sys
from os.path import join, basename, splitext
import glob
import numpy as np
import torch
import argparse
import PIL
from torchvision.transforms import ToTensor
from imageio import imread

from model import *
from utils.image_utils import *
from utils.utils import *

import cv2

from collections import OrderedDict

iscuda = torch.cuda.is_available()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', required=True, type=str,
                        help="ImageNetVID root Directory")
    parser.add_argument('-m', '--model', required=False, default='Baseline', type=str,
                        help="Model Name")
    parser.add_argument('-w', '--weight', required=True, type=str,
                        help="Model Weight Folder")
    parser.add_argument('-g', '--gpu', required=False, type=str, default='False',
                        help="Using GPU")

    args = parser.parse_args()
    if args.gpu == 'True':
        args.gpu = True
    else:
        args.gpu = False

    if args.gpu and not iscuda:
        args.gpu = False
    return args


def remove_moudule(model_dict):
    ret = OrderedDict()
    for k, v in model_dict.items():
        name = k[7:]
        ret[name] = v
    return ret

def show_image(image, bbox, waitkey=1):
    image = cv2.cvtColor(np.uint8(image), cv2.COLOR_RGB2BGR)
    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0,0,255), 1)

    cv2.imshow("show", image)
    return cv2.waitKey(waitkey)


def show_patch(image, bbox):
    image = cv2.cvtColor(np.uint8(image), cv2.COLOR_RGB2BGR)
    img = image[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2], :]
    cv2.imshow("Patch", img)
    cv2.waitKey(10)


def update_bbox(bbox, dy, dx, scale, influence):
    bbox[0] = int(bbox[0] * (1 - influence) + (bbox[0] + dx * scale) * influence)
    bbox[1] = int(bbox[1] * (1 - influence) + (bbox[1] + dy * scale) * influence)

    bbox[2] = int(bbox[2]*(1-influence)) + int(bbox[2]*scale*influence)
    bbox[3] = int(bbox[3]*(1-influence)) + int(bbox[3]*scale*influence)
    return bbox


def get_seq_list(root_dir):
    glob_expression = os.path.join(root_dir, 'Data', 'VID', 'val', "*")
    rel_path = [os.path.relpath(path, join(root_dir,'Data', 'VID', 'val')) for path in sorted(glob.glob(glob_expression))]
    return rel_path


def get_image_list(seq, root_dir):
    seq_path = join(root_dir, 'Data', 'VID', 'val', seq, '*.JPEG')
    image_list = [basename(path) for path in sorted(glob.glob(seq_path))]
    return image_list


def get_first_bbox(image_path, seq_path, image_dir, annot_dir, cxt_margin, num_of_scale=5, transform=ToTensor()):
    path = join(image_dir, seq_path, image_path)
    img = np.float32(imread(path))
    annot, wid, hei, isvalid = get_annotations(annot_dir, seq_path, image_path)
    if not isvalid:
        return None, None, None, None
    ref_wid = int(annot['xmax'] - annot['xmin'])
    ref_hei = int(annot['ymax'] - annot['ymin'])
    cxt = cxt_margin*(ref_wid + ref_hei)
    ref_sz = math.sqrt((ref_wid + cxt)*(ref_hei+cxt))
    ref_sz = ref_sz//2 * 2 + 1
    cx = int((annot['xmax'] + annot['xmin'])/2)
    cy = int((annot['ymax'] + annot['ymin'])/2)
    ret_img = []
    for i in range(num_of_scale):
        cropped_img, pad = crop_img(img, cy, cx, ref_sz)
        ret_img.append(transform(resize_and_pad(cropped_img, 127, pad, ref_sz, True, resize)))
    bbox = [annot['xmin'], annot['ymin'], ref_wid, ref_hei]
    return torch.stack(ret_img), bbox, ref_sz, img


def get_srch_image(image_path, seq_path, image_dir, annot_dir, bbox, ref_cxt_size, ref_size, srch_size, scale_factors, srch_ctx_size=None, transform=ToTensor()):
    path = join(image_dir, seq_path, image_path)
    srch_img = imread(path)
    annot, wid, hei, isvalid = get_annotations(annot_dir, seq_path, image_path)
    ret_img = []
    for scale in scale_factors:
        if srch_ctx_size is None:
            srch_ctx_size = ref_cxt_size * srch_size / ref_size
            scale_srch_ctx_size = srch_ctx_size*scale // 2 * 2 + 1
        else:
            scale_srch_ctx_size = srch_ctx_size*scale // 2 * 2 + 1
        cx = int(bbox[0]*scale + bbox[2]*scale // 2 + 1)
        cy = int(bbox[1]*scale + bbox[3]*scale // 2 + 1)
        rz_srch_img = resize(srch_img, (int(wid*scale), int(hei*scale)))
        cropped, pad = crop_img(np.float32(rz_srch_img), cy, cx, scale_srch_ctx_size)
        ret_img.append(transform(resize_and_pad(cropped, srch_size, pad, scale_srch_ctx_size, True, resize)))
    return torch.stack(ret_img), annot, srch_img, srch_ctx_size


# def update_ref_image(image, bbox, num_of_scales, ctx_margin, transform=ToTensor()):
#     wid = bbox[2]
#     hei = bbox[3]
#     cx = bbox[0] + bbox[2]//2
#     cy = bbox[1] + bbox[1]//2
#     ctx = ctx_margin*(wid + hei)
#     ref_sz = math.sqrt((wid+ ctx)*(hei+ctx))
#     ref_sz = ref_sz//2 * 2 + 1
#     ret_img = []
#     for i in range(num_of_scales):
#         cropped_img, pad = crop_img(image, cx, cy, ref_sz)
#         ret_img.append(transform(resize_and_pad(cropped_img, 127, pad, ref_sz, True, resize)))
#
#     return torch.stack(ret_img)
#
def clac_center_error(annotation, detected):
    return None



def main(args):
    param = dict(scale_num=3, scale_step=1.04, window_influence=0.35, response_up=8,
                 scale_min=0.2, scale_max=5, final_score_sz=33, cxt_margin=0.5, ctx_influence=.59)
    assert check_imagenet_folder_tree(args.path), "Check ImageNetVID Folder"
    if args.gpu:
        device = 'cuda'
    else:
        device = 'cpu'
    siamfc = SiameseNet(Baseline(), 'cos', param['final_score_sz'], param['response_up']).to(device)
    loaded_dict = torch.load(args.weight)
    # loaded_dict = torch.load(args.weight, map_location={'cuda:2': 'cuda:0'})
    model_dict = remove_moudule(loaded_dict['model'])
    siamfc.load_state_dict(model_dict)
    image_dir = join(args.path, 'Data', 'VID', 'val')
    annot_dir = join(args.path, 'Annotations', 'VID', 'val')
    seq_list = get_seq_list(args.path)
    scale_factor = param['scale_step']**(np.linspace(-np.ceil(param['scale_num']/2), np.ceil(param['scale_num']/2), param['scale_num']))
    scale_penalty = 0.95
    hann_1d = np.expand_dims(np.hanning(255), axis=0)
    penalty = np.transpose(hann_1d) * hann_1d
    # penalty = np.ones_like(penalty)
    # penalty = penalty / np.sum(penalty)
    for seq in seq_list:
        image_list = get_image_list(seq, args.path)
        ref_img, bbox, ref_cxt_sz, ref_origin = get_first_bbox(image_list[0], seq, image_dir, annot_dir, 0.5, len(scale_factor))
        if ref_img is None:
            continue
        show_patch(ref_origin, bbox)
        srch_ctx_size = None
        for img_path in image_list:
            srch_img, srch_bbox, srch_origin, srch_ctx_size = get_srch_image(img_path, seq, image_dir, annot_dir, bbox, ref_cxt_sz,
                                                              127, 255, scale_factor, srch_ctx_size)
            score_map = siamfc(ref_img.to(device), srch_img.to(device))
            score_map = score_map.detach().cpu().permute(0, 2, 3, 1)
            scores = []
            for idx, score in enumerate(score_map):
                scores.append(np.multiply(cv2.resize(score.numpy(), (255, 255), interpolation=cv2.INTER_CUBIC), penalty) * (scale_penalty**abs(idx - int(len(scale_factor)/2))))
            max_idx = np.argmax(np.asarray(scores).reshape((-1, 1, 1)), axis=0)
            max= np.max(np.asarray(scores).reshape((-1, 1, 1)), axis=0)
            scale_index = max_idx // 255**2
            dy = (max_idx - scale_index*(255**2)) // 255 - 127
            dx = (max_idx - scale_index*(255**2)) % 255 - 127
            print(max)

            bbox = update_bbox(bbox, int(dy), int(dx), scale_factor[scale_index], param['window_influence'])
            srch_ctx_size = srch_ctx_size*(1.-param['ctx_influence']) + srch_ctx_size*scale_factor[scale_index]*param['ctx_influence']
            key_val = show_image(srch_origin, bbox, 1)
            if key_val == 27:
                break


if __name__ == '__main__':
   sys.argv += '-p D:/Dataset/ILSVRC2015_VID/ILSVRC2015 -w output/0001.pt -g True'.split(' ')
   args = parse_arguments()
   main(args)

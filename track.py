import os, sys
from os.path import join, basename, splitext
import glob
import cv2
import numpy as np
import torch
import argparse
import PIL
from torchvision.transforms import ToTensor
from imageio import imread

from model import *
from utils.image_utils import *
from utils.utils import *

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


def main(args):
    param = dict(scale_num=3, scale_step=1.04, window_influence=0.25, response_up=8,
                 scale_min=0.2, scale_max=5)

    assert check_imagenet_folder_tree(args.path), "Check ImageNetVID Folder"
    embedding_net = Baseline()
    if args.gpu:
        device = 'cuda'
    else:
        device = 'cpu'
    siamfc = SiameseNet(embedding_net, 'cos').to(device)
    loaded_dict = torch.load(args.weight)
    # siamnet.load_state_dict(loaded_dict['model'])
    image_dir = join(args.path, 'Data', 'VID', 'val')
    annot_dir = join(args.path, 'Annotations', 'VID', 'val')
    seq_list = get_seq_list(args.path)
    scale_factor = param['scale_step']**(np.linspace(-np.ceil(param['scale_num']/2)), np.ceil(param['scale_num']/2), param['scale_num'])
    scale_penalty = 0.97
    hann_1d = np.expand_dims(np.hanning(255), axis=0)
    penalty = np.transpose(hann_1d) * hann_1d
    penalty = penalty / np.sum(penalty)
    avg_scale = RunningAverage()

    for seq in seq_list:
        first, image_list = get_image_list(seq, args.path)
        ref_img, bbox, ref_cxt_sz, ref_origin = get_first_bbox(first, seq, image_dir, annot_dir, 0.5, 127, len(scale_factor))
        if ref_img is None:
            continue
        show_patch(ref_origin, bbox)
        for img_path in image_list:
            srch_img, srch_bbox, srch_origin = get_srch_image(img_path, seq, image_dir, annot_dir, bbox, ref_cxt_sz, 127, 255, scale_factor)
            score_map = siamfc(ref_img.to(device), srch_img.to(device))
            score_map = score_map.detach().cpu().permute(0, 2, 3, 1)
            scores = []
            for idx, score in enumerate(score_map):
                scores.append(np.multiply(cv2.resize(score.numpy(), (255,255), interpolation=cv2.INTER_CUBIC), penalty)*scale_penelty)
            max_idx = np.argmax(np.asarray(scores).reshape((-1, 1, 1)), axis=0)
            scale_index = max_idx // 255**2
            dy = (max_idx - scale_index*(255**2)) // 255 - 127
            dx = (max_idx - scale_index*(255**2)) % 255 - 127
            avg_scale.update(float(scale_factor[scale_index]))
            bbox = update_bbox(bbox, int(dy), int(dx), avg_scale())
            key_val = show_image(srch_origin, bbox)
            if key_val == 27:
                break


def show_image(image, bbox):
    #image has to convert bgr layer
    image = cv2.cvtColor(np.uint8(image), cv2.COLOR_RGB2BGR)
    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0,0,255), 1)
    # cv2.circle(image, (bbox[0] + bbox[2]//2, bbox[1] + bbox[3]//2), 2, (0,0,255),2)
    cv2.imshow("show", image)
    return cv2.waitKey(1)


def show_patch(image, bbox):
    image = cv2.cvtColor(np.uint8(image), cv2.COLOR_RGB2BGR)
    img = image[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2], :]
    cv2.imshow("Patch", img)
    cv2.waitKey(10)


def update_bbox(bbox, dy, dx, scale):
    bbox[0] = int(round(bbox[0] + dx*scale))
    bbox[1] = int(round(bbox[1] + dy*scale))

    bbox[2] = int(round(bbox[2]*scale))
    bbox[3] = int(round(bbox[3]*scale))
    return bbox


def get_seq_list(root_dir):
    glob_expression = os.path.join(root_dir, 'Data', 'VID', 'val', "*")
    rel_path = [os.path.relpath(path, join(root_dir,'Data', 'VID', 'val')) for path in sorted(glob.glob(glob_expression))]
    return rel_path


def get_image_list(seq, root_dir):
    seq_path = join(root_dir, 'Data', 'VID', 'val', seq, '*.JPEG')
    image_list = [basename(path) for path in sorted(glob.glob(seq_path))]
    return image_list[0], image_list[1:]


def get_first_bbox(image_path, seq_path, image_dir, annot_dir, cxt_margin, ref_size, num_of_scale=5, transform=ToTensor()):
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
        ret_img.append(transform(resize_and_pad(cropped_img, ref_size, pad, ref_sz, True, resize)))
    bbox = [annot['xmin'], annot['ymin'], ref_wid, ref_hei]
    return torch.stack(ret_img), bbox, ref_sz, img


def get_srch_image(image_path, seq_path, image_dir, annot_dir,bbox, ref_cxt_size, ref_size, srch_size, scale_factors, transform=ToTensor()):
    path = join(image_dir, seq_path, image_path)
    srch_img = imread(path)
    annot, wid, hei, isvalid = get_annotations(annot_dir, seq_path, image_path)
    srch_ctx_size = ref_cxt_size * srch_size/ref_size
    srch_ctx_size = srch_ctx_size // 2 * 2 + 1
    ret_img = []
    for scale in scale_factors:
        cx = int(bbox[0]*scale + bbox[2]*scale // 2)
        cy = int(bbox[1]*scale + bbox[3]*scale // 2)
        rz_srch_img = resize(srch_img, (int(wid*scale), int(hei*scale)))
        cropped, pad = crop_img(np.float32(rz_srch_img), cy, cx, srch_ctx_size)
        ret_img.append(transform(resize_and_pad(cropped, srch_size, pad, srch_ctx_size, True, resize)))
    return torch.stack(ret_img), annot, srch_img


if __name__ == '__main__':
   sys.argv += '-p D:/Dataset/ILSVRC2015_VID/ILSVRC2015 -w output/0003.pt -g True'.split(' ')
   args = parse_arguments()
   main(args)

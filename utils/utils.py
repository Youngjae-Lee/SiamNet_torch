import os
from os.path import splitext, isfile, join
import numpy as np
from xml.etree import ElementTree as ET
import json
import torch
import torch.nn.functional as F
import time


class RunningAverage(object):
    def __init__(self):
        self.total = 0
        self.count = 0

    def update(self, val):
        self.total += val
        self.count += 1

    def get(self):
        return {
            "total": self.total,
            "count": self.count
        }

    def set(self, val):
        self.total += val['total']
        self.count += val['count']

    def __call__(self, *args, **kwargs):
        return self.total / float(self.count)


class RunningAverageMultiVar:
    def __init__(self, **kwargs):
        self.avg_dict = dict()
        for key, val in kwargs.items():
            self.avg_dict.update({key: val})

    def update(self, **kwargs):
        for key, val in kwargs.items():
            self.avg_dict[key].update(val)

    def __getitem__(self, item):
        return self.avg_dict[item]

    def get(self, key):
        return self.avg_dict[key]()


class Params:
    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def update_with_dict(self, dictio):
        self.__dict__.update(dictio)

    @property
    def dict(self):
        return self.__dict__


def get_annotations(annot_dir, sequence_dir, frame_file):
    frame_number = splitext(frame_file)[0]
    annot_file = join(annot_dir, sequence_dir, frame_number+'.xml')
    if isfile(annot_file):
        tree = ET.parse(annot_file)
        root = tree.getroot()
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        if root.find('object') is None:
            annotation = {'xmax':None, 'ymax':None, 'xmin':None, 'ymin':None}
            valid_frame = False
            return annotation, width, height, valid_frame
        track_id_zero_present = False
        for obj in root.findall('object'):
            if obj.find('trackid').text == '0':
                bbox = obj.find('bndbox')
                xmax = int(bbox.find('xmax').text)
                xmin = int(bbox.find('xmin').text)
                ymax = int(bbox.find('ymax').text)
                ymin = int(bbox.find('ymin').text)
                track_id_zero_present = True
            if not track_id_zero_present:
                annotation = {'xmax':None, 'ymax':None, 'xmin':None, 'ymin':None}
                valid_frame = False
                return annotation, width, height, valid_frame
    else:
        raise FileNotFoundError("File not founded")

    annotation = {'xmax': xmax, 'xmin':xmin, 'ymin':ymin, 'ymax':ymax}
    valid_frame = True

    return annotation, width, height, valid_frame


def check_imagenet_folder_tree(root_dir):
    data_type = ['Annotations', 'Data']
    dataset_type = ['train', 'val']
    folders = [join(root_dir, data, 'VID', dataset) for data in data_type for dataset in dataset_type]
    return all(os.path.isdir(path) for path in folders)


def save_model(path_to_save, save_params):
    assert save_params is not None, "Running Parameter have to exist"
    curr_epoch = save_params.get("epoch")
    output_name = "{0:04d}".format(curr_epoch)
    model_output = output_name + ".pt"
    model_output = join(path_to_save, model_output)
    torch.save(save_params, model_output)


def load_model(path_to_load, model, optim, scheduler, param):
    ckpt_path = path_to_load + ".pt"
    try:
        dict = torch.load(ckpt_path)
    except FileNotFoundError:
        print("That file not exit")
        return
    model.load_state_dict(dict['model'])
    scheduler.load_state_dict(dict['scheduler'])
    optim.load_state_dict(dict['optim'])
    param.start_epoch = dict['epoch']

    return model, optim, scheduler


def cosine_similarity(ref, srch, batchnorm=None):
    b, c, h, w = srch.shape
    srch_view = srch.view(1, b*c, h, w)
    score_map = F.conv2d(srch_view, ref, groups=b)
    score_map = score_map.permute(1, 0, 2, 3)

    # one_vector = torch.ones_like(ref, dtype=torch.float32, requires_grad=False)
    # sum_vec = F.conv2d(torch.pow(srch_view, 2), one_vector, groups=b, )
    # sum_vec = sum_vec.permute(1, 0, 2, 3)
    #
    # sqrt_A = torch.sqrt(torch.sum(torch.pow(ref, 2), dim=[1,2,3], keepdim=True))
    # sqrt_B = torch.sqrt(sum_vec)
    # sqrt_A = sqrt_A.expand_as(sqrt_B)
    # score_map = score_map / (sqrt_A*sqrt_B)

    if batchnorm is not None:
        score_map = batchnorm(score_map)
    return score_map


def min_max_similarity(ref, srch, batchnorm=None):
    b, c, h, w = srch.shape
    srch_view = srch.view(1, b*c, h, w)
    score_map = F.conv2d(srch_view, ref, groups=b)
    score_map = score_map.permute(1, 0, 2, 3)
    if batchnorm is not None:
        score_map = batchnorm(score_map)
    # score_map = (score_map - min) / (max - min)
    return score_map


SIM_DICT = {
    "cos": cosine_similarity,
    "minmax": min_max_similarity
}
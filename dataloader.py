import os
from os.path import join, relpath, isfile, isdir, basename, splitext
from math import sqrt
import random
import glob
import tqdm
import json
import cv2

import numpy as np
from imageio import imread
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset

from labels import *
from utils.utils import *
from utils.image_utils import *


class ImageNetVID(Dataset):
    def __init__(self, imagenet_dir, transform=ToTensor(),
                 reference_size=127, search_size=255, final_size=33,
                 lable_fcn=None, upscale_factor=4, max_frame_seq=50, pos_thr=25, neg_thr=50,
                 cxt_margin=.5, single_label=True, img_read_fcn=imread, resize_fcn=None,
                 metadata_file=None, save_metadata=None):
        if not check_imagenet_folder_tree(imagenet_dir):
            raise FileNotFoundError("error")
        self.set_root_dir(imagenet_dir)
        self.max_frame_seq = max_frame_seq
        self.reference_size = reference_size
        self.search_size = search_size
        self.upscale_factor = upscale_factor
        self.cxt_margin = cxt_margin
        self.final_size = final_size
        self.pos_thr = pos_thr
        self.neg_thr = neg_thr
        self.transform = transform
        self.label_func = lable_fcn
        if single_label:
            self.label = self.label_func(self.final_size, self.pos_thr, self.neg_thr, self.upscale_factor)
        else:
            self.label = None
        self.imread = img_read_fcn
        self.resize = resize_fcn
        self.frames = dict()
        self.annotations = dict()
        self.get_metadata(metadata_file, save_metadata)

    def set_root_dir(self, root_dir):
        self.image_dir = join(root_dir, 'Data', 'VID', 'train')
        self.annot_dir = join(root_dir, 'Annotations', 'VID', 'train')

    def __getitem__(self, idx):
        seq_idx = self.list_idx[idx]
        first_idx, second_idx = self.get_pairs(seq_idx)
        return self.process_sample(seq_idx, first_idx, second_idx)

    def __len__(self):
        return len(self.list_idx)

    def get_metadata(self, metadata_file=None, save_metadata=None):
        if metadata_file and isfile(metadata_file):
            with open(metadata_file) as json_file:
                mdata = json.load(json_file)
                if self.check_metadata(mdata):
                    for key, value in mdata.items():
                        setattr(self, key, value)
                    return
        mdata = self.build_metadata()
        if save_metadata is not None:
            with open(save_metadata, 'w') as outfile:
                json.dump(mdata, outfile, indent=3)

    def check_metadata(self, metadata):
        if not all(key in metadata for key in ('frames', 'annotations', 'list_idx')):
            return False
        if not (isfile(metadata['frames'][0][0]) and isfile(metadata['frames'][-1][-1])):
            return False
        return True

    def get_pairs(self, seq_idx, frame_idx=None):
        size = len(self.frames[seq_idx])
        if frame_idx is None:
            first_frame_idx = random.randint(0, size-1)
        else:
            first_frame_idx = frame_idx

        min_frame_idx = max(0, (first_frame_idx - self.max_frame_seq))
        max_frame_idx = min(size-1, (first_frame_idx + self.max_frame_seq))
        second_frame_idx = random.randint(min_frame_idx, max_frame_idx)

        return first_frame_idx, second_frame_idx

    def ref_context_size(self, h, w):
        margin_size = self.cxt_margin*(w + h)
        ref_size = sqrt((w+margin_size) * (h+margin_size))
        ref_size = (ref_size//2)*2 + 1
        return int(ref_size)

    def process_sample(self, seq_idx, first_idx, second_idx):
        ref_frame_path = self.frames[seq_idx][first_idx]
        ref_annot = self.annotations[seq_idx][first_idx]

        srch_frame_path = self.frames[seq_idx][second_idx]
        srch_annot = self.annotations[seq_idx][second_idx]

        ref_w = (ref_annot['xmax'] - ref_annot['xmin'])
        ref_h = (ref_annot['ymax'] - ref_annot['ymin'])
        ref_ctx_size = self.ref_context_size(ref_h, ref_w)
        ref_cx = int((ref_annot['xmax'] + ref_annot['xmin'])/2)
        ref_cy = int((ref_annot['ymax'] + ref_annot['ymin'])/2)

        ref_frame = self.imread(ref_frame_path)
        ref_frame = np.float32(ref_frame)

        ref_frame, pad_amounts_ref = crop_img(ref_frame, ref_cy, ref_cx, ref_ctx_size)
        try:
            ref_frame = resize_and_pad(ref_frame, self.reference_size, pad_amounts_ref,
                                       ref_ctx_size, use_avg=True,
                                       resize_func=self.resize)
        except AssertionError:
            print("error")
            raise

        srch_ctx_size = ref_ctx_size * self.search_size / self.reference_size
        srch_ctx_size = (srch_ctx_size//2)*2 + 1

        srch_cx = (srch_annot['xmax'] + srch_annot['xmin'])/2
        srch_cy = (srch_annot['ymax'] + srch_annot['ymin'])/2

        srch_frame = self.imread(srch_frame_path)
        srch_frame = np.float32(srch_frame)
        srch_frame, pad_amounts_srch = crop_img(srch_frame, srch_cy, srch_cx, srch_ctx_size)
        try:
            srch_frame = resize_and_pad(srch_frame, self.search_size, pad_amounts_srch, srch_ctx_size,
                                        use_avg=True, resize_func=self.resize)
        except AssertionError:
            raise

        if self.label is not None:
            label = self.label
        else:
            label = self.label_func(self.final_size, self.pos_thr, self.neg_thr,
                                    upscale_factor=self.upscale_factor)

        ref_frame = self.transform(ref_frame)
        srch_frame = self.transform(srch_frame)

        out_dict = {'ref':ref_frame, 'srch':srch_frame, 'label': label,
                    'seq_idx':seq_idx, 'ref_idx': first_idx, 'srch_idx':second_idx}

        return out_dict

    def build_metadata(self):
        frames = []
        annotations = []
        list_idx = []

        sequence_dirs = self.get_scenes_dirs()
        for idx, sequence in enumerate(tqdm.tqdm(sequence_dirs)):
            seq_frames = []
            seq_annotations = []
            for frame in sorted(os.listdir(join(self.image_dir, sequence))):
                annot, h, w, valid = get_annotations(self.annot_dir, sequence, frame)
                if valid:
                    seq_frames.append(join(self.image_dir, sequence, frame))
                    seq_annotations.append(annot)
                    list_idx.append(idx)
            frames.append(seq_frames)
            annotations.append(seq_annotations)

        metadata = {'frames': frames, 'annotations': annotations, 'list_idx': list_idx}

        for key, value in metadata.items():
            setattr(self, key, value)

        return metadata

    def get_scenes_dirs(self):
        glob_expression = join(self.image_dir, '*', '*')
        relative_paths = [relpath(p, self.image_dir) for p in sorted(glob.glob(glob_expression))]
        return relative_paths


def get_train_metafile(imageset_dir, num_of_train_data=-1):
    path = join(imageset_dir, 'VID', '*.txt')
    files = glob.glob(path)
    train_list = []
    for f in files:
        name = splitext(basename(f))[0].split('_')
        if name[0] == 'train':
            train_list.append(f)

    if num_of_train_data != -1:
        random.shuffle(train_list)
        train_list = train_list[:num_of_train_data]
        return train_list
    else:
        return train_list


class ImageNetVID_val(ImageNetVID):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_root_dir(self, root_dir):
        self.image_dir = join(root_dir, 'Data', 'VID', 'val')
        self.annot_dir = join(root_dir, 'Annotations', 'VID', 'val')

    def get_scenes_dirs(self):
        glob_expression = join(self.image_dir, '*')
        relative_paths = [relpath(p, self.image_dir) for p in sorted(glob.glob(glob_expression))]
        return relative_paths

    def check_metadata(self, metadata):
        if not super().check_metadata(metadata):
            return False
        if not all(key in metadata for key in ('list_pairs', 'max_frame_seq')):
            return False

        if metadata['max_frame_seq'] != self.max_frame_seq:
            return False
        return True

    def build_metadata(self):
        metatdata = super().build_metadata()
        self.list_pairs = self.build_test_pairs()
        metatdata['list_pairs'] = self.list_pairs
        metatdata['max_frame_seq'] = self.max_frame_seq
        return metatdata

    def build_test_pairs(self):
        random.seed(100)
        list_pairs = []
        for seq_idx, seq in enumerate(self.frames):
            for frame_idx in range(len(seq)):
                list_pairs.append([seq_idx, *super().get_pairs(seq_idx, frame_idx)])
        random.shuffle(list_pairs)
        random.seed()
        return list_pairs

    def __getitem__(self, idx):
        list_idx, first_idx, second_idx = self.list_pairs[idx]
        return self.process_sample(list_idx, first_idx, second_idx)


if __name__ == '__main__':
    dataset = ImageNetVID("D:/Dataset/ILSVRC2015_VID/ILSVRC2015",
                          lable_fcn=create_BCELogit_loss_label,
                          metadata_file='meta/meta.json',
                          img_read_fcn=imread,
                          resize_fcn=resize)
    loader = torch.utils.data.DataLoader(dataset, 1, False)
    for idx, sample in enumerate(loader):
        continue


""" A dataset parser that reads images from folders

Folders are scannerd recursively to find image files. Labels are based
on the folder hierarchy, just leaf folders by default.

Hacked together by / Copyright 2020 Ross Wightman
"""
import os
import json

from timm.utils.misc import natural_key

from .parser import Parser
from .class_map import load_class_map
from .constants import IMG_EXTENSIONS


def find_images_and_targets(root, json_file, types=IMG_EXTENSIONS, class_to_idx=None, leaf_name_only=True, sort=True):
    labels = []
    filenames = []
    with open(json_file, 'r') as f:
        j_obj = json.load(f)
        j_imgs = j_obj['imagenet']

    for img_sample in j_imgs:
        img_name = img_sample['img_info']['filename']
        label = img_sample['annos']['imagenet']['classification']
        base, ext = os.path.splitext(img_name)
        if ext.lower() in types:
            filenames.append(os.path.join(root, img_name))
            labels.append(label)

    images_and_targets = [(f, l) for f, l in zip(filenames, labels)]
    if sort:
        images_and_targets = sorted(images_and_targets, key=lambda k: natural_key(k[0]))
    return images_and_targets


class ParserImageJson(Parser):

    def __init__(
            self,
            root,
            json_file=None,
            use_cache=False):
        super().__init__()

        self.root = root
        self.samples = find_images_and_targets(root, json_file=json_file)
        self.use_cache = use_cache
        if len(self.samples) == 0:
            raise RuntimeError(
                f'Found 0 images in subfolders of {root}. Supported image extensions are {", ".join(IMG_EXTENSIONS)}')

    def __getitem__(self, index):
        path, target = self.samples[index]
        if not self.use_cache:
            return open(path, 'rb'), target
        else:
            return path, target

    def __len__(self):
        return len(self.samples)

    def _filename(self, index, basename=False, absolute=False):
        filename = self.samples[index][0]
        if basename:
            filename = os.path.basename(filename)
        elif not absolute:
            filename = os.path.relpath(filename, self.root)
        return filename

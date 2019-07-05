from __future__ import print_function, absolute_import
import os.path as osp
import numpy as np
import pdb
from glob import glob
import re
class DA(object):
    def __init__(self, data_dir, source, target,source_train_path,target_train_path, source_extension, target_extension):
        # source / target image root
        self.source_images_dir = osp.join(data_dir, source)
        self.target_images_dir = osp.join(data_dir, target)
        self.source_extension = source_extension
        self.target_extension = target_extension
        self.source_train_path = source_train_path # dir origanized : data_dir/source/source_train_path
        self.target_train_path = target_train_path # dir origanized : data_dir/target/target_train_path
        self.gallery_path = 'bounding_box_test'
        self.query_path = 'query'

        self.source_train, self.target_train, self.query, self.gallery = [], [], [], []
        self.num_train_ids, self.num_query_ids, self.num_gallery_ids = 0, 0, 0

        self.cam_dict = self.set_cam_dict()
        self.target_num_cam = self.cam_dict[target]
        self.source_num_cam = self.cam_dict[source]

        self.load()

    def set_cam_dict(self):
        cam_dict = {}
        cam_dict['market'] = 6
        cam_dict['duke'] = 8
        cam_dict['msmt17'] = 15
        cam_dict['market2s01'] = 6
        cam_dict['s012market'] = 6
        cam_dict['sys'] = 2
        return cam_dict


    def preprocess(self, images_dir, path, img_extension='jpg', relabel=True):
        pattern = re.compile(r'([-\d]+)_c([-\d]+)')
        all_pids = {}
        ret = []
        img_extension = '*.' + img_extension
        fpaths = sorted(glob(osp.join(images_dir,path,img_extension)))
        for fpath in fpaths:
            fname = osp.basename(fpath)
            if 'cuhk03' in images_dir:
                name = osp.splitext(fname)[0]
                pid, cam = map(int, pattern.search(fname).groups())
            else:
                pid, cam = map(int, pattern.search(fname).groups())
            if pid == -1: continue  # junk images are just ignored
            if relabel:
                if pid not in all_pids:
                    all_pids[pid] = len(all_pids)
            else:
                if pid not in all_pids:
                    all_pids[pid] = pid
            pid = all_pids[pid]
            cam -= 1
            ret.append((fname, pid, cam))
        return ret, int(len(all_pids))


    def load(self):
        self.source_train, self.num_source_train_ids = self.preprocess(self.source_images_dir, self.source_train_path,img_extension=self.source_extension)
        self.target_train, self.num_target_train_ids = self.preprocess(self.target_images_dir, self.target_train_path,img_extension=self.target_extension)
        self.query, self.num_query_ids = self.preprocess(self.target_images_dir, self.query_path, img_extension=self.target_extension, relabel=False)
        self.gallery, self.num_gallery_ids = self.preprocess(self.target_images_dir, self.gallery_path, img_extension=self.target_extension, relabel=False)
        print(self.__class__.__name__, "dataset loaded")
        print("  subset   | # ids | # images")
        print("  ---------------------------")
        print("  source train    | {:5d} | {:8d}"
              .format(self.num_source_train_ids, len(self.source_train)))
        print("  target train    | {:5d} | {:8d}"
              .format(self.num_target_train_ids, len(self.target_train)))
        print("  query           | {:5d} | {:8d}"
              .format(self.num_query_ids,len(self.query)))
        print("  gallery           | {:5d} | {:8d}"
              .format(self.num_gallery_ids,len(self.gallery)))

        print('source path : %s' % str(self.source_images_dir))
        print('target path : %s' % str(self.target_images_dir))
        print('query path : %s' % str(osp.join(self.target_images_dir,self.query_path)))
        print('gallery path : %s' % str(osp.join(self.target_images_dir,self.gallery_path)))




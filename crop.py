import numpy as np
import cv2
import argparse
import os
import json
def crop(ori_img,nb_img):
    y,x = np.where((nb_img[:,:,0] * nb_img[:,:,1] * nb_img[:,:,2]) != 0)
    y_min = np.min(y)
    y_max = np.max(y)
    x_min = np.min(x)
    x_max = np.max(x)
    w = x_max - x_min
    h = y_max - y_min
    bbox = [str(x_min),str(y_min),str(w),str(h)]
    cp = ori_img[y_min:y_max,x_min:x_max,:]
    cp_nb = nb_img[y_min:y_max,x_min:x_max,:]
    return cp,cp_nb,bbox
def main(args):
    ori_img_root = args.ori_img_root
    nb_img_root = args.nb_img_root
    ori_ds = os.listdir(ori_img_root)
    nb_ds = os.listdir(nb_img_root)
    for ori_d in ori_ds:
        ori_img_dir = os.path.join(ori_img_root,ori_d)
        nb_img_dir = os.path.join(nb_img_root,ori_d)
        id = ori_d
        bbf_name = id + '_bbox.txt'
        bbf = os.path.join(ori_img_dir,bbf_name)
        if not os.path.exists(bbf):
            os.mknod(bbf)
        ori_imgs = os.listdir(ori_img_dir)
        for ori_img in ori_imgs:
            if '.png' not in ori_img:
                continue
            cur_img_path = os.path.join(ori_img_dir,ori_img)
            cur_nb_path = os.path.join(nb_img_dir,ori_img)
            cur_ori = cv2.imread(cur_img_path)
            cur_nb = cv2.imread(cur_nb_path)
            cp,cp_nb,bbx = crop(cur_ori,cur_nb)
            cv2.imwrite(cur_img_path,cp)
            cv2.imwrite(cur_nb_path,cp_nb)
            with open(bbf,'a') as f:
                f.write(ori_img)
                f.write(' '.join(bbx))
                f.write('\n')






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ori_img_root',type=str,default='/ssd4/ltb/datasets/PersonX/test/ori')
    parser.add_argument('--nb_img_root',type=str,default='/ssd4/ltb/datasets/PersonX/test/nb')
    args = parser.parse_args()
    main(args)
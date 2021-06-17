#!/usr/bin/env python
"""
    File Name   :   CoSeg-scanNet3D
    date        :   14/10/2019
    Author      :   wenbo
    Email       :   huwenbodut@gmail.com
    Description :
                              _     _
                             ( |---/ )
                              ) . . (
________________________,--._(___Y___)_,--._______________________
                        `--'           `--'
"""

import torch.utils.data as data
import torch
import numpy as np
from os.path import join, exists
from glob import glob
import multiprocessing as mp
import SharedArray as SA
import dataset.augmentation as t
from dataset.voxelizer import Voxelizer


def sa_create(name, var):
    x = SA.create(name, var.shape, dtype=var.dtype)
    x[...] = var[...]
    x.flags.writeable = False
    return x


def collation_fn(batch):
    """
    :param batch:
    :return:   coords_batch: N x 4 (x,y,z,batch)

    """
    coords, feats, labels = list(zip(*batch))

    for i in range(len(coords)):
        coords[i][:, 0] *= i

    return torch.cat(coords), torch.cat(feats), torch.cat(labels)


def collation_fn_eval_all(batch):
    """
    :param batch:
    :return:   coords_batch: N x 4 (x,y,z,batch)

    """
    coords, feats, labels, inds_recons = list(zip(*batch))
    inds_recons = list(inds_recons)
    # pdb.set_trace()

    accmulate_points_num = 0
    for i in range(len(coords)):
        coords[i][:, 0] *= i
        inds_recons[i] = accmulate_points_num + inds_recons[i]
        accmulate_points_num += coords[i].shape[0]

    return torch.cat(coords), torch.cat(feats), torch.cat(labels), torch.cat(inds_recons)


class ScanNet3D(data.Dataset):
    # Augmentation arguments
    SCALE_AUGMENTATION_BOUND = (0.9, 1.1)
    ROTATION_AUGMENTATION_BOUND = ((-np.pi / 64, np.pi / 64), (-np.pi / 64, np.pi / 64), (-np.pi,
                                                                                          np.pi))
    TRANSLATION_AUGMENTATION_RATIO_BOUND = ((-0.2, 0.2), (-0.2, 0.2), (0, 0))
    ELASTIC_DISTORT_PARAMS = ((0.2, 0.4), (0.8, 1.6))

    ROTATION_AXIS = 'z'
    LOCFEAT_IDX = 2

    def __init__(self, dataPathPrefix='Data', voxelSize=0.05,
                 split='train', aug=False, memCacheInit=False, identifier=1233, loop=1,
                 data_aug_color_trans_ratio=0.1, data_aug_color_jitter_std=0.05, data_aug_hue_max=0.5,
                 data_aug_saturation_max=0.2, eval_all=False
                 ):
        super(ScanNet3D, self).__init__()
        self.split = split
        self.identifier = identifier
        self.data_paths = sorted(glob(join(dataPathPrefix, split, '*.pth')))
        self.voxelSize = voxelSize
        self.aug = aug
        self.loop = loop
        self.eval_all = eval_all

        self.voxelizer = Voxelizer(
            voxel_size=voxelSize,
            clip_bound=None,
            use_augmentation=True,
            scale_augmentation_bound=self.SCALE_AUGMENTATION_BOUND,
            rotation_augmentation_bound=self.ROTATION_AUGMENTATION_BOUND,
            translation_augmentation_ratio_bound=self.TRANSLATION_AUGMENTATION_RATIO_BOUND)

        if aug:
            prevoxel_transform_train = [t.ElasticDistortion(self.ELASTIC_DISTORT_PARAMS)]
            self.prevoxel_transforms = t.Compose(prevoxel_transform_train)
            input_transforms = [
                t.RandomHorizontalFlip(self.ROTATION_AXIS, is_temporal=False),
                t.ChromaticAutoContrast(),
                t.ChromaticTranslation(data_aug_color_trans_ratio),
                t.ChromaticJitter(data_aug_color_jitter_std),
                t.HueSaturationTranslation(data_aug_hue_max, data_aug_saturation_max),
            ]
            self.input_transforms = t.Compose(input_transforms)

        if memCacheInit and (not exists("/dev/shm/wbhu_scannet_3d_%s_%06d_locs_%08d" % (split, identifier, 0))):
            print('[*] Starting shared memory init ...')
            for i, (locs, feats, labels) in enumerate(torch.utils.data.DataLoader(
                    self.data_paths, collate_fn=lambda x: torch.load(x[0]),
                    num_workers=min(8, mp.cpu_count()), shuffle=False)):
                labels[labels == -100] = 255
                labels = labels.astype(np.uint8)
                # Scale color to 0-255
                feats = (feats + 1.) * 127.5
                sa_create("shm://wbhu_scannet_3d_%s_%06d_locs_%08d" % (split, identifier, i), locs)
                sa_create("shm://wbhu_scannet_3d_%s_%06d_feats_%08d" % (split, identifier, i), feats)
                sa_create("shm://wbhu_scannet_3d_%s_%06d_labels_%08d" % (split, identifier, i), labels)

        print('[*] %s (%s) loading done (%d)! ' % (dataPathPrefix, split, len(self.data_paths)))

    def __getitem__(self, index_long):
        index = index_long % len(self.data_paths)
        locs_in = SA.attach("shm://wbhu_scannet_3d_%s_%06d_locs_%08d" % (self.split, self.identifier, index)).copy()
        feats_in = SA.attach("shm://wbhu_scannet_3d_%s_%06d_feats_%08d" % (self.split, self.identifier, index)).copy()
        labels_in = SA.attach("shm://wbhu_scannet_3d_%s_%06d_labels_%08d" % (self.split, self.identifier, index)).copy()

        locs = self.prevoxel_transforms(locs_in) if self.aug else locs_in
        locs, feats, labels, inds_reconstruct = self.voxelizer.voxelize(locs, feats_in, labels_in)
        if self.eval_all:
            labels = labels_in
        if self.aug:
            locs, feats, labels = self.input_transforms(locs, feats, labels)
        coords = torch.from_numpy(locs).int()
        coords = torch.cat((torch.ones(coords.shape[0], 1, dtype=torch.int), coords), dim=1)
        feats = torch.from_numpy(feats).float() / 127.5 - 1.
        labels = torch.from_numpy(labels).long()

        if self.eval_all:
            return coords, feats, labels, torch.from_numpy(inds_reconstruct).long()
        return coords, feats, labels

    def __len__(self):
        return len(self.data_paths) * self.loop


if __name__ == '__main__':
    import time, random
    from tensorboardX import SummaryWriter

    data_root = '/research/dept6/wbhu/Dataset/ScanNet'
    train_data = ScanNet3D(dataPathPrefix=data_root, aug=True, split='train', memCacheInit=True, voxelSize=0.05)
    val_data = ScanNet3D(dataPathPrefix=data_root, aug=False, split='val', memCacheInit=True, voxelSize=0.05,
                         eval_all=True)

    manual_seed = 123


    def worker_init_fn(worker_id):
        random.seed(manual_seed + worker_id)


    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True, num_workers=4, pin_memory=True,
                                               worker_init_fn=worker_init_fn, collate_fn=collation_fn)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=16, shuffle=False, num_workers=4, pin_memory=True,
                                             worker_init_fn=worker_init_fn, collate_fn=collation_fn_eval_all)
    trainLog = SummaryWriter('Exp/scannet/statistic/train')
    valLog = SummaryWriter('Exp/scannet/statistic/val')

    for idx in range(1):
        end = time.time()
        for step, (coords, feat, label) in enumerate(train_loader):
            print(
                'time: {}/{}--{}'.format(step + 1, len(train_loader), time.time() - end))
            trainLog.add_histogram('voxel_coord_x', coords[:, 0], global_step=step)
            trainLog.add_histogram('voxel_coord_y', coords[:, 1], global_step=step)
            trainLog.add_histogram('voxel_coord_z', coords[:, 2], global_step=step)
            trainLog.add_histogram('color', feat, global_step=step)
            # time.sleep(0.3)
            end = time.time()

        for step, (coords, feat, label, inds_reverse) in enumerate(val_loader):
            print(
                'time: {}/{}--{}'.format(step + 1, len(val_loader), time.time() - end))
            valLog.add_histogram('voxel_coord_x', coords[:, 0], global_step=step)
            valLog.add_histogram('voxel_coord_y', coords[:, 1], global_step=step)
            valLog.add_histogram('voxel_coord_z', coords[:, 2], global_step=step)
            valLog.add_histogram('color', feat, global_step=step)
            # time.sleep(0.3)
            end = time.time()

    trainLog.close()
    valLog.close()

"""
This file is adapted from https://github.com/hugoycj/Instant-angelo
"""

import os
import math
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torchvision.transforms.functional as TF

import pytorch_lightning as pl

import datasets
from datasets.colmap_utils import \
    read_cameras_binary, read_images_binary
from models.ray_utils import get_ray_directions
from utils.misc import get_rank
from datasets.sparsecraft_utils import ply_to_point_cloud
from torch_cluster import knn


def get_center(pts):
    center = pts.mean(0)
    dis = (pts - center[None,:]).norm(p=2, dim=-1)
    mean, std = dis.mean(), dis.std()
    q25, q75 = torch.quantile(dis, 0.25), torch.quantile(dis, 0.75)
    valid = (dis > mean - 1.5 * std) & (dis < mean + 1.5 * std) & (dis > mean - (q75 - q25) * 1.5) & (dis < mean + (q75 - q25) * 1.5)
    center = pts[valid].mean(0)
    return center

def normalize_poses(poses, pts, up_est_method, center_est_method, pts3d_normal=None):
    if center_est_method == 'camera':
        # estimation scene center as the average of all camera positions
        center = poses[...,3].mean(0)
    elif center_est_method == 'lookat':
        # estimation scene center as the average of the intersection of selected pairs of camera rays
        cams_ori = poses[...,3]
        cams_dir = poses[:,:3,:3] @ torch.as_tensor([0.,0.,-1.])
        cams_dir = F.normalize(cams_dir, dim=-1)
        A = torch.stack([cams_dir, -cams_dir.roll(1,0)], dim=-1)
        b = -cams_ori + cams_ori.roll(1,0)
        t = torch.linalg.lstsq(A, b).solution
        center = (torch.stack([cams_dir, cams_dir.roll(1,0)], dim=-1) * t[:,None,:] + torch.stack([cams_ori, cams_ori.roll(1,0)], dim=-1)).mean((0,2))
    elif center_est_method == 'point':
        # first estimation scene center as the average of all camera positions
        # later we'll use the center of all points bounded by the cameras as the final scene center
        center = poses[...,3].mean(0)
    else:
        raise NotImplementedError(f'Unknown center estimation method: {center_est_method}')

    if up_est_method == 'ground':
        # estimate up direction as the normal of the estimated ground plane
        # use RANSAC to estimate the ground plane in the point cloud
        import pyransac3d as pyrsc
        ground = pyrsc.Plane()
        plane_eq, inliers = ground.fit(pts.numpy(), thresh=0.01) # TODO: determine thresh based on scene scale
        plane_eq = torch.as_tensor(plane_eq) # A, B, C, D in Ax + By + Cz + D = 0
        z = F.normalize(plane_eq[:3], dim=-1) # plane normal as up direction
        signed_distance = (torch.cat([pts, torch.ones_like(pts[...,0:1])], dim=-1) * plane_eq).sum(-1)
        if signed_distance.mean() < 0:
            z = -z # flip the direction if points lie under the plane
    elif up_est_method == 'camera':
        # estimate up direction as the average of all camera up directions
        z = F.normalize((poses[...,3] - center).mean(0), dim=0)
    else:
        raise NotImplementedError(f'Unknown up estimation method: {up_est_method}')

    # new axis
    y_ = torch.as_tensor([z[1], -z[0], 0.])
    x = F.normalize(y_.cross(z), dim=0)
    y = z.cross(x)

    if center_est_method == 'point':
        # rotation
        Rc = torch.stack([x, y, z], dim=1)
        R = Rc.T
        poses_homo = torch.cat([poses, torch.as_tensor([[[0.,0.,0.,1.]]]).expand(poses.shape[0], -1, -1)], dim=1)
        inv_trans = torch.cat([torch.cat([R, torch.as_tensor([[0.,0.,0.]]).T], dim=1), torch.as_tensor([[0.,0.,0.,1.]])], dim=0)
        poses_norm = (inv_trans @ poses_homo)[:,:3]
        pts = (inv_trans @ torch.cat([pts, torch.ones_like(pts[:,0:1])], dim=-1)[...,None])[:,:3,0]

        # translation and scaling
        poses_min, poses_max = poses_norm[...,3].min(0)[0], poses_norm[...,3].max(0)[0]
        pts_fg = pts[(poses_min[0] < pts[:,0]) & (pts[:,0] < poses_max[0]) & (poses_min[1] < pts[:,1]) & (pts[:,1] < poses_max[1])]
        center = get_center(pts_fg)
        tc = center.reshape(3, 1)
        t = -tc
        poses_homo = torch.cat([poses_norm, torch.as_tensor([[[0.,0.,0.,1.]]]).expand(poses_norm.shape[0], -1, -1)], dim=1)
        inv_trans = torch.cat([torch.cat([torch.eye(3), t], dim=1), torch.as_tensor([[0.,0.,0.,1.]])], dim=0)
        poses_norm = (inv_trans @ poses_homo)[:,:3]
        scale = poses_norm[...,3].norm(p=2, dim=-1).min()
        poses_norm[...,3] /= scale
        pts_trans = (inv_trans @ torch.cat([pts, torch.ones_like(pts[:,0:1])], dim=-1)[...,None])[:,:3,0]
        # apply the rotation to the point cloud normal
        if pts3d_normal is not None:
            pts3d_normal = (R @ pts3d_normal.T).T
        pts_trans = pts_trans / scale
    else:
        # rotation and translation
        Rc = torch.stack([x, y, z], dim=1)
        tc = center.reshape(3, 1)
        R, t = Rc.T, -Rc.T @ tc
        poses_homo = torch.cat([poses, torch.as_tensor([[[0.,0.,0.,1.]]]).expand(poses.shape[0], -1, -1)], dim=1)
        inv_trans = torch.cat([torch.cat([R, t], dim=1), torch.as_tensor([[0.,0.,0.,1.]])], dim=0)
        poses_norm = (inv_trans @ poses_homo)[:,:3] # (N_images, 4, 4)

        # scaling
        scale = poses_norm[...,3].norm(p=2, dim=-1).min()
        poses_norm[...,3] /= scale
        # apply the transformation to the point cloud
        pts_trans = (inv_trans @ torch.cat([pts, torch.ones_like(pts[:,0:1])], dim=-1)[...,None])[:,:3,0]
        # apply the rotation to the point cloud normal
        if pts3d_normal is not None:
            pts3d_normal = (R @ pts3d_normal.T).T
        pts_trans = pts_trans / scale

        # orig_pts = pts_trans * scale
        # trans = torch.tensor(np.linalg.inv(inv_trans))

        # orig_pts = (trans @ torch.cat([orig_pts, torch.ones_like(orig_pts[:,0:1])], dim=-1)[...,None])[:,:3,0]

    return poses_norm, pts_trans, pts3d_normal

def create_spheric_poses(cameras, n_steps=120):
    center = torch.as_tensor([0.,0.,0.], dtype=cameras.dtype, device=cameras.device)
    mean_d = (cameras - center[None,:]).norm(p=2, dim=-1).mean()
    mean_h = cameras[:,2].mean()
    r = (mean_d**2 - mean_h**2).sqrt()
    up = torch.as_tensor([0., 0., 1.], dtype=center.dtype, device=center.device)

    all_c2w = []
    for theta in torch.linspace(0, 2 * math.pi, n_steps):
        cam_pos = torch.stack([r * theta.cos(), r * theta.sin(), mean_h])
        l = F.normalize(center - cam_pos, p=2, dim=0)
        s = F.normalize(l.cross(up), p=2, dim=0)
        u = F.normalize(s.cross(l), p=2, dim=0)
        c2w = torch.cat([torch.stack([s, u, -l], dim=1), cam_pos[:,None]], axis=1)
        all_c2w.append(c2w)

    all_c2w = torch.stack(all_c2w, dim=0)
    
    return all_c2w

def round_python3(number):
    rounded = round(number)
    if abs(number - rounded) == 0.5:
        return 2.0 * round(number / 2.0)
    return rounded

from models.utils import scale_anything
from nerfacc import ContractionType
def contract_to_unisphere(x, radius, contraction_type):
    if contraction_type == ContractionType.AABB:
        x = scale_anything(x, (-radius, radius), (0, 1))
    elif contraction_type == ContractionType.UN_BOUNDED_SPHERE:
        x = scale_anything(x, (-radius, radius), (0, 1))
        x = x * 2 - 1  # aabb is at [-1, 1]
        mag = x.norm(dim=-1, keepdim=True)
        mask = mag.squeeze(-1) > 1
        x[mask] = (2 - 1 / mag[mask]) * (x[mask] / mag[mask])
        x = x / 4 + 0.5  # [-inf, inf] is at [0, 1]
    else:
        raise NotImplementedError
    return x

def error_to_confidence(error):
    # Here smaller_beta means slower transition from 0 to 1.
    # Increasing beta raises steepness of the curve.
    beta = 1
    # confidence = 1 / np.exp(beta*error)
    confidence = 1 / (1 + np.exp(beta*error))
    return confidence

class ColmapDatasetBase():
    # the data only has to be processed once
    initialized = False
    properties = {}

    def setup(self, config, split):
        self.config = config
        self.split = split
        self.rank = get_rank()
        self.radius = self.config["radius"]

        if not ColmapDatasetBase.initialized:
            camdata = read_cameras_binary(os.path.join(self.config.root_dir, 'sparse/cameras.bin'))

            H = int(camdata[1].height)
            W = int(camdata[1].width)

            if 'img_wh' in self.config:
                w, h = self.config.img_wh
                assert round(W / w * h) == H
            elif 'img_downscale' in self.config:
                w, h = int(W / self.config.img_downscale + 0.5), int(H / self.config.img_downscale + 0.5)
            else:
                raise KeyError("Either img_wh or img_downscale should be specified.")

            img_wh = (w, h)
            factor = w / W

            if camdata[1].model == 'SIMPLE_RADIAL':
                fx = fy = camdata[1].params[0] * factor
                cx = camdata[1].params[1] * factor
                cy = camdata[1].params[2] * factor
            elif camdata[1].model in ['PINHOLE', 'OPENCV']:
                fx = camdata[1].params[0] * factor
                fy = camdata[1].params[1] * factor
                cx = camdata[1].params[2] * factor
                cy = camdata[1].params[3] * factor
            elif camdata[1].model == 'SIMPLE_PINHOLE':
                fx = fy = camdata[1].params[0] * factor
                cx = camdata[1].params[1] * factor
                cy = camdata[1].params[2] * factor
            else:
                raise ValueError(f"Please parse the intrinsics for camera model {camdata[1].model}!")
            
            directions = get_ray_directions(w, h, fx, fy, cx, cy)

            imdata = read_images_binary(os.path.join(self.config.root_dir, 'sparse/images.bin'))

            mask_dir = os.path.join(self.config.root_dir, 'mask')
            has_mask = os.path.exists(mask_dir)
            apply_mask = has_mask and self.config.apply_mask
            
            all_c2w, all_images, all_fg_masks, all_paths = [], [], [], []
            all_fg_indexs, all_bg_indexs = [], []
            for i, d in enumerate(imdata.values()):
                R = d.qvec2rotmat()
                t = d.tvec.reshape(3, 1)
                c2w = torch.from_numpy(np.concatenate([R.T, -R.T@t], axis=1)).float()
                c2w[:,1:3] *= -1. # COLMAP => OpenGL
                all_c2w.append(c2w)
                if self.split in ['train', 'val']:
                    img_path = os.path.join(self.config.root_dir, 'images', d.name)
                    img = Image.open(img_path)
                    img = img.resize(img_wh, Image.BICUBIC)
                    img = TF.to_tensor(img).permute(1, 2, 0)[...,:3]
                    if has_mask:
                        mask_paths = [os.path.join(mask_dir, d.name), os.path.join(mask_dir, d.name[3:])]
                        mask_paths = list(filter(os.path.exists, mask_paths))
                        assert len(mask_paths) == 1
                        mask = Image.open(mask_paths[0]).convert('L') # (H, W, 1)
                        mask = mask.resize(img_wh, Image.BICUBIC)
                        mask = TF.to_tensor(mask)[0]
                    else:
                        mask = torch.ones_like(img[...,0], device=img.device)
                    fg_index = torch.stack(torch.nonzero(mask.bool(), as_tuple=True), dim=0)
                    bg_index = torch.stack(torch.nonzero(~mask.bool(), as_tuple=True), dim=0)
                    fg_index = torch.cat([torch.full((1, fg_index.shape[1]), i), fg_index], dim=0)
                    bg_index = torch.cat([torch.full((1, bg_index.shape[1]), i), bg_index], dim=0)
                    all_fg_indexs.append(fg_index.permute(1, 0))
                    all_bg_indexs.append(bg_index.permute(1, 0))
                    all_fg_masks.append(mask) # (h, w)
                    all_images.append(img)
                    all_paths.append(img_path)
            
            all_c2w = torch.stack(all_c2w, dim=0)   

            pcl_file = self.config.get("pcl_file", None)
            if pcl_file:
                pts3d, pts_colors, pts_normals = ply_to_point_cloud(
                    os.path.join(self.config.root_dir, pcl_file), self.radius
                )
                # Normalize the poses and the point cloud
                pts3d = torch.from_numpy(pts3d).float()
                all_c2w, pts3d, pts_normals = normalize_poses(all_c2w, pts3d, up_est_method=self.config.up_est_method, center_est_method=self.config.center_est_method, pts3d_normal=pts_normals)
                
                self.pts_colors = torch.from_numpy(pts_colors).float().to(self.rank)
                self.pts_normals = pts_normals.float().to(self.rank)
                self.pts_normals = F.normalize(self.pts_normals, dim=-1)
                self.pts3d = pts3d.to(self.rank)
                print(
                    f"Loaded point cloud from file containing {self.pts3d.shape[0]} points"
                )
                # Pre-compute these values once
                self.POINT_NUM_GT = self.pts3d.shape[0]
                self.dim = self.pts3d.shape[1]
                
                if self.split == "train":
                    self.num_queries = 5

                    # Pre-allocate tensors
                    self.point_idx = torch.empty(self.POINT_NUM_GT, dtype=torch.long, device=self.rank)
                    self.sample = torch.empty((self.num_queries, self.POINT_NUM_GT, self.dim), device=self.rank)
                    self.n_idx = torch.empty((self.num_queries * self.POINT_NUM_GT, 1), dtype=torch.long, device=self.rank)

                    self.generate_taylor_samples(sigma=0.01)
            else:
                raise NotImplementedError("Please provide a point cloud file!")

            

            ColmapDatasetBase.properties = {
                'w': w,
                'h': h,
                'img_wh': img_wh,
                'factor': factor,
                'has_mask': has_mask,
                'apply_mask': apply_mask,
                'directions': directions,
                'all_c2w': all_c2w,
                'all_images': all_images,
                'all_paths': all_paths,
                'all_fg_masks': all_fg_masks,
                'all_fg_indexs': all_fg_indexs,
                'all_bg_indexs': all_bg_indexs
            }

            ColmapDatasetBase.initialized = True
        
        for k, v in ColmapDatasetBase.properties.items():
            setattr(self, k, v)

        image_ids = list(range(len(self.all_images)))
        if self.split == "train":
            image_ids = image_ids
        elif self.split == "val":
            image_ids = image_ids[:1]
        elif self.split == "test":
            # sample 1/10 of the images for testing uniformly
            idx_sub = [round_python3(i) for i in np.linspace(0, len(image_ids)-1, int(len(image_ids) * 0.1))]
            image_ids = [c for idx, c in enumerate(image_ids) if idx in idx_sub]
            

        self.all_images, self.all_fg_masks = torch.stack(self.all_images, dim=0).float(), torch.stack(self.all_fg_masks, dim=0).float()
        self.all_fg_indexs = torch.cat(self.all_fg_indexs, dim=0)
        self.all_bg_indexs = torch.cat(self.all_bg_indexs, dim=0)

        self.all_c2w = self.all_c2w[[image_ids]].float()
        self.all_images = self.all_images[image_ids]
        self.all_paths = [self.all_paths[i] for i in image_ids]
        self.all_fg_masks = self.all_fg_masks[image_ids]

        print(f"Number of views for {self.split} split is {len(self.all_images)}")


    def generate_taylor_samples(self, sigma=0.01):
        max_dist = 0.25 * sigma

        self.point_idx = torch.randperm(self.POINT_NUM_GT, device=self.rank)[:self.POINT_NUM_GT]
        
        pointcloud = self.pts3d[self.point_idx]
        normals = self.pts_normals[self.point_idx]
        colors = self.pts_colors[self.point_idx]

        # Generate samples using PyTorch operations
        self.sample = pointcloud.unsqueeze(0) + max_dist * torch.randn(
            (self.num_queries, self.POINT_NUM_GT, self.dim), device=self.rank
        )

        _, self.n_idx = knn(pointcloud, self.sample.view(-1, self.dim), k=1)

        sample_near = pointcloud[self.n_idx].view(self.num_queries, self.POINT_NUM_GT, self.dim)
        normals_near = normals[self.n_idx].view(self.num_queries, self.POINT_NUM_GT, self.dim)
        colors_near = colors[self.n_idx].view(self.num_queries, self.POINT_NUM_GT, self.dim)

        self.input_points = sample_near.reshape(-1, 3)
        self.input_normals = normals_near.reshape(-1, 3)
        self.input_colors = colors_near.reshape(-1, 3)
        self.query_points = self.sample.reshape(-1, 3).clamp(-self.radius, self.radius)

        return self.input_points, self.input_normals, self.input_colors, self.query_points


class ColmapDataset(Dataset, ColmapDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, index):
        return {"index": index}


class ColmapIterableDataset(IterableDataset, ColmapDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __iter__(self):
        while True:
            yield {}


@datasets.register('colmap')
class ColmapDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        if stage in [None, 'fit']:
            self.train_dataset = ColmapIterableDataset(self.config, 'train')
        if stage in [None, 'fit', 'validate']:
            self.val_dataset = ColmapDataset(self.config, self.config.get('val_split', 'valid'))
        if stage in [None, 'test']:
            self.test_dataset = ColmapDataset(self.config, self.config.get('test_split', 'test'))
        if stage in [None, 'predict']:
            self.predict_dataset = ColmapDataset(self.config, 'train')         

    def prepare_data(self):
        pass
    
    def general_loader(self, dataset, batch_size):
        sampler = None
        return DataLoader(
            dataset, 
            num_workers=os.cpu_count(), 
            batch_size=batch_size,
            pin_memory=True,
            sampler=sampler
        )
    
    def train_dataloader(self):
        return self.general_loader(self.train_dataset, batch_size=1)

    def val_dataloader(self):
        return self.general_loader(self.val_dataset, batch_size=1)

    def test_dataloader(self):
        return self.general_loader(self.test_dataset, batch_size=1) 

    def predict_dataloader(self):
        return self.general_loader(self.predict_dataset, batch_size=1)       

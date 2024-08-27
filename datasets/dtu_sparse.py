import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torchvision.transforms.functional as TF

import pytorch_lightning as pl

import datasets
from models.ray_utils import get_ray_directions
from utils.misc import get_rank
from datasets.sparsecraft_utils import ply_to_point_cloud, load_K_Rt_from_P
import cv2
from torch_cluster import knn


class DTUDatasetBase:
    def setup(self, config, split):
        self.config = config
        self.split = split
        self.rank = get_rank()
        self.radius = self.config["radius"]
        cams = np.load(os.path.join(self.config.root_dir, self.config.cameras_file))
        # get one image as a sample
        images_in_dir = os.listdir(os.path.join(self.config.root_dir, "image"))
        image_sample = images_in_dir[0]
        img_sample = cv2.imread(
            os.path.join(self.config.root_dir, "image", image_sample)
        )
        H, W = img_sample.shape[0], img_sample.shape[1]

        if "img_wh" in self.config:
            w, h = self.config.img_wh
            assert round(W / w * h) == H
        elif "img_downscale" in self.config:
            w, h = int(W / self.config.img_downscale + 0.5), int(
                H / self.config.img_downscale + 0.5
            )
            print(f"Images are downscaled by a factor of: {self.config.img_downscale}")
        else:
            w, h = W, H
        #    raise KeyError("Either img_wh or img_downscale should be specified.")

        pcl_file = self.config.get("pcl_file", None)
        if pcl_file:
            pts3d, pts_colors, pts_normals = ply_to_point_cloud(
                os.path.join(self.config.root_dir, pcl_file), self.radius
            )
            self.pts_colors = torch.from_numpy(pts_colors).float().to(self.rank)
            self.pts_normals = torch.from_numpy(pts_normals).float().to(self.rank)
            self.pts_normals = F.normalize(self.pts_normals, dim=-1)
            self.pts3d = torch.from_numpy(pts3d).float().to(self.rank)
            print(
                f"Loaded point cloud from file containing {self.pts3d.shape[0]} points"
            )
            # Pre-compute these values once
            self.POINT_NUM_GT = self.pts3d.shape[0]
            self.dim = self.pts3d.shape[1]
            
            if self.split == "train":
                if self.config.views_n <= 6:
                    self.num_queries = 10
                else:
                    self.num_queries = 3

                # Pre-allocate tensors
                self.point_idx = torch.empty(self.POINT_NUM_GT, dtype=torch.long, device=self.rank)
                self.sample = torch.empty((self.num_queries, self.POINT_NUM_GT, self.dim), device=self.rank)
                self.n_idx = torch.empty((self.num_queries * self.POINT_NUM_GT, 1), dtype=torch.long, device=self.rank)

                self.generate_taylor_samples(sigma=0.01)
        self.w, self.h = w, h
        self.img_wh = (w, h)
        self.img_hw = (h, w)
        self.factor = w / W
        mask_dir = os.path.join(self.config.root_dir, "mask")
        self.has_mask = True
        self.apply_mask = self.config.apply_mask

        self.directions = []
        self.all_intrinsics = []

        (
            self.all_c2w,
            self.all_k,
            self.all_images,
            self.all_fg_masks,
            self.all_paths,
            self.all_w2c,
        ) = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        
        # select indices based on the task
        if self.config.task == "sparse_novel_view_synthesis":
            train_idx = [25, 22, 28, 40, 44, 48, 0, 8, 13]
            exclude_idx = [3, 4, 5, 6, 7, 16, 17,
                           18, 19, 20, 21, 36, 37, 38, 39]
            test_idx = [i for i in np.arange(
                49) if i not in train_idx + exclude_idx]
            indices = train_idx if self.split == "train" else test_idx
            if self.split == "train" and self.config.views_n > 0:
                indices = indices[: self.config.views_n]
        elif self.config.task == "surface_reconstruction":
            indices = []
            for e in images_in_dir:
                img_id = int(e.split(".")[0])
                indices.append(img_id)
        else:
            
            raise KeyError("Task type must be specified.")

        print(f"Using images {indices} for split {self.split}")
        for i in indices:
            img_path = os.path.join(self.config.root_dir, "image", f"{i:06d}.png")
            world_mat, scale_mat = cams[f"world_mat_{i}"], cams[f"scale_mat_{i}"]
            P = (world_mat @ scale_mat)[:3, :4]
            K, c2w = load_K_Rt_from_P(P)
            fx, fy, cx, cy = (
                K[0, 0] * self.factor,
                K[1, 1] * self.factor,
                K[0, 2] * self.factor,
                K[1, 2] * self.factor,
            )
            directions = get_ray_directions(w, h, fx, fy, cx, cy)
            self.directions.append(directions)
            c2w = torch.from_numpy(c2w).float()
            # blender follows opengl camera coordinates (right up back)
            # NeuS DTU data coordinate system (right down front) is different from blender
            # https://github.com/Totoro97/NeuS/issues/9
            # for c2w, flip the sign of input camera coordinate yz
            c2w_ = c2w.clone()
            c2w_[:3, 1:3] *= -1.0  # flip input sign
            c2w_opengl = c2w_[:3, :4]
            self.all_c2w.append(c2w_opengl)
            self.all_w2c.append(torch.inverse(c2w))
            self.all_k.append(torch.from_numpy(K).float())
            self.all_paths.append(img_path)
            img = Image.open(img_path)
            img = img.resize(self.img_wh, Image.BICUBIC)
            img = TF.to_tensor(img).permute(1, 2, 0)[..., :3]

            # Masks are only available for testing
            if self.split != "train":
                mask_path = os.path.join(mask_dir, f"{i:03d}.png")
                mask = Image.open(mask_path).convert("L")  # (H, W, 1)
                mask = mask.resize(self.img_wh, Image.BICUBIC)
                mask = TF.to_tensor(mask)[0]
                self.all_fg_masks.append(mask)  # (h, w)
            self.all_images.append(img)
        self.all_c2w = torch.stack(self.all_c2w, dim=0)
        self.all_w2c = torch.stack(self.all_w2c, dim=0)
        self.all_k = torch.stack(self.all_k, dim=0)
        self.all_images = torch.stack(self.all_images, dim=0)
        self.directions = torch.stack(self.directions, dim=0)

        self.directions = self.directions.float().to(self.rank)
        self.all_c2w, self.all_images, self.all_k, self.all_w2c = (
            self.all_c2w.float().to(self.rank),
            self.all_images.float().to(self.rank),
            self.all_k.float().to(self.rank),
            self.all_w2c.float().to(self.rank),
        )
        if self.split != "train":
            self.all_fg_masks = (
                torch.stack(self.all_fg_masks, dim=0).float().to(self.rank)
            )
        
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


class DTUDataset(Dataset, DTUDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, index):
        return {"index": index}


class DTUIterableDataset(IterableDataset, DTUDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __iter__(self):
        while True:
            yield {}


@datasets.register("dtu")
class DTUDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        if stage in [None, "fit"]:
            self.train_dataset = DTUIterableDataset(self.config, "train")
        if stage in [None, "fit", "validate"]:
            self.val_dataset = DTUDataset(
                self.config, self.config.get("val_split", "valid")
            )
        if stage in [None, "test"]:
            self.test_dataset = DTUDataset(
                self.config, self.config.get("test_split", "test")
            )
        if stage in [None, "predict"]:
            self.predict_dataset = DTUDataset(self.config, "train")

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size):
        sampler = None
        return DataLoader(
            dataset,
            num_workers=os.cpu_count(),
            batch_size=batch_size,
            pin_memory=True,
            sampler=sampler,
        )

    def train_dataloader(self):
        return self.general_loader(self.train_dataset, batch_size=1)

    def val_dataloader(self):
        return self.general_loader(self.val_dataset, batch_size=1)

    def test_dataloader(self):
        return self.general_loader(self.test_dataset, batch_size=1)

    def predict_dataloader(self):
        return self.general_loader(self.predict_dataset, batch_size=1)

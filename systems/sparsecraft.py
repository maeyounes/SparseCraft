import systems
from systems.neus import NeuSSystem
import torch.nn.functional as F
import torch
import numpy as np


@systems.register("sparsecraft-system")
class SparseCraftSystem(NeuSSystem):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, batch):
        return self.model(batch["rays"])

    def training_step(self, batch, batch_idx):
        out = self(batch)
        loss = 0.0
        # NOTE: Update train_num_rays for dynamic ray sampling training
        if self.config.model.dynamic_ray_sampling:
            train_num_rays = int(
                self.train_num_rays
                * (self.train_num_samples / out["num_samples_full"].sum().item())
            )
            self.train_num_rays = min(
                int(self.train_num_rays * 0.9 + train_num_rays * 0.1),
                self.config.model.max_train_num_rays,
            )
        # NOTE: Eikonal regularization: Enforce the norm of sdf gradients to be 1
        loss_eikonal = (
            (torch.linalg.norm(out["sdf_grad_samples"], ord=2, dim=-1) - 1.0) ** 2
        ).mean()
        self.log("train/loss_eikonal", loss_eikonal)
        loss += loss_eikonal * self.C(self.config.system.loss.lambda_eikonal)
        # NOTE: Taylor based Regurlarization with MVS points
        if self.C(self.config.system.loss.lambda_input_taylor) or self.C(self.config.system.loss.lambda_query_taylor):
            # Update the query points to match epsilon of numerical diff
            if self.global_step % 500 == 0:
                self.dataset.generate_taylor_samples(sigma=self.model.geometry._finite_difference_eps * 2.)
            # Sample input and query points
            if self.dataset.input_points.shape[0] > 5000:
                points_samples_ids = np.random.choice(self.dataset.input_points.shape[0], 5000, replace=False)
                batch_input_points = self.dataset.input_points[points_samples_ids]
                batch_input_normals = self.dataset.input_normals[points_samples_ids]
                batch_input_colors = self.dataset.input_colors[points_samples_ids]
                batch_query_points = self.dataset.query_points[points_samples_ids]
            else:
                batch_input_points = self.dataset.input_points
                batch_input_normals = self.dataset.input_normals
                batch_input_colors = self.dataset.input_colors
                batch_query_points = self.dataset.query_points
            # NOTE: Apply Taylor expansion of first order at the input points
            # f(p) = f(q) + dfdq * (p - q)  <=>  p = q - (f(q)/||dfdq||²)* dfdq
            # Query sdf values and gradients for query points
            query_sdf, query_sdf_g = self.model.geometry(
                    batch_query_points, with_grad=True, with_feature=False
            )
            sum_square_grad = torch.sum(
                query_sdf_g ** 2, dim=-1, keepdim=True)
            batch_moved_points = batch_query_points.detach() - (query_sdf_g / sum_square_grad) * query_sdf.unsqueeze(-1)
            input_taylor_loss = torch.linalg.norm((batch_input_points - batch_moved_points), ord=2, dim=-1).mean()
            loss += input_taylor_loss * self.C(self.config.system.loss.lambda_input_taylor)
            self.log(
                    "train/input_taylor_loss", input_taylor_loss.detach()
            )
            if self.C(self.config.system.loss.lambda_query_taylor) or self.C(self.config.system.loss.lambda_mvs_color):
                # NOTE: Apply Taylor expansion of first order at the query points
                # f(q) = f(p) + dfdp * (q - p) => f(q) = ||dfdq||² * n (q - p) <=> p = q - f(q) * ||dfdp||² * n 
                # we omit the term ||dfdp||² as it enforced by the eikonal constraint to be 1
                _, input_sdf_g, input_feature, input_encoding = self.model.geometry(
                    batch_input_points, with_grad=True, with_feature=True, with_encoding=True
                )
                batch_moved_q = batch_query_points.detach() -  query_sdf.unsqueeze(-1) * batch_input_normals
                query_taylor_loss = (torch.linalg.norm((batch_input_points - batch_moved_q), ord=2, dim=-1)).mean()
                loss += query_taylor_loss * self.C(self.config.system.loss.lambda_query_taylor)
                self.log(
                    "train/query_taylor_loss", query_taylor_loss.detach()
                )
            if self.C(self.config.system.loss.lambda_mvs_color):
                # NOTE: Apply mvs albedo color loss
                input_mvs_albedo = self.model.texture.forward_albedo(input_encoding, input_feature)
                loss_input_mvs_albedo_l1 = F.l1_loss(input_mvs_albedo, batch_input_colors)
                loss += loss_input_mvs_albedo_l1 * self.C(
                    self.config.system.loss.lambda_mvs_color
                )
                self.log("train/mvs_color_l1_loss", loss_input_mvs_albedo_l1.detach())
        # NOTE: RGB loss
        loss_rgb_l1 = F.l1_loss(out["comp_rgb_full"], batch["rgb"])
        self.log("train/loss_rgb", loss_rgb_l1)
        loss += loss_rgb_l1 * self.C(self.config.system.loss.lambda_rgb_l1)
        # NOTE: Specular regularization: at the beginning of the training, the capacity of the model is limited, it should not explain color with specular effects
        if self.C(self.config.system.loss.lambda_specular_color):
            specular_color = out["specular_color"]
            loss_specular_color = specular_color.abs().mean()
            loss += loss_specular_color * self.C(self.config.system.loss.lambda_specular_color)
            self.log("train/loss_specular_color", loss_specular_color)
        self.log("train/inv_s", out["inv_s"], prog_bar=True)
        for name, value in self.config.system.loss.items():
            if name.startswith("lambda"):
                self.log(f"train_params/{name}", self.C(value))

        self.log("train/num_rays", float(self.train_num_rays), prog_bar=True)
        self.log("train/global_loss", loss)
        return {"loss": loss}


    def validation_step(self, batch, batch_idx):
        out = self(batch)
        psnr = self.criterions["psnr"](
            out["comp_rgb_full"].to(batch["rgb"]), batch["rgb"]
        )
        W, H = self.dataset.img_wh
        self.save_image_grid(
            f"it{self.global_step}-{batch['index'][0].item()}.png",
            [
                {
                    "type": "rgb",
                    "img": batch["rgb"].view(H, W, 3),
                    "kwargs": {"data_format": "HWC"},
                },
                {
                    "type": "rgb",
                    "img": out["comp_rgb_full"].view(H, W, 3),
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb_bg"].view(H, W, 3),
                        "kwargs": {"data_format": "HWC"},
                    },
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"].view(H, W, 3),
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if self.config.model.learned_background
                else []
            )
            + [
                {"type": "grayscale", "img": out["depth"].view(H, W), "kwargs": {}},
                {
                    "type": "rgb",
                    "img": out["comp_normal"].view(H, W, 3),
                    "kwargs": {"data_format": "HWC", "data_range": (-1, 1)},
                },
            ],
        )
        # compute masked psnr if dataset has mask
        if self.dataset.has_mask:
            masked_psnr = self.criterions["psnr"](
                out["comp_rgb_full"].to(batch["rgb"]),
                batch["rgb"],
                valid_mask=batch["fg_mask"].bool(),
            )
            return {"psnr": psnr, "masked_psnr": masked_psnr, "index": batch["index"]}
        return {"psnr": psnr, "index": batch["index"]}
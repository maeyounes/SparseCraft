import torch
import torch.nn.functional as F

from models.ray_utils import get_rays
import systems
from systems.base import BaseSystem
from systems.criterions import (
    PSNR,
    SSIM,
    LPIPS,
    compute_avg_error,
)


@systems.register("neus-system")
class NeuSSystem(BaseSystem):
    """
    Two ways to print to console:
    1. self.print: correctly handle progress bar
    2. rank_zero_info: use the logging module
    """

    def prepare(self):
        if self.training:
            self.criterions = {"psnr": PSNR()}
        else:
            self.criterions = {"psnr": PSNR(), "ssim": SSIM(), "lpips": LPIPS()}
        self.train_num_samples = self.config.model.train_num_rays * (
            self.config.model.num_samples_per_ray
            + self.config.model.get("num_samples_per_ray_bg", 0)
        )
        self.train_num_rays = self.config.model.train_num_rays

    def on_test_start(self):
        self.criterions = {"psnr": PSNR(), "ssim": SSIM(), "lpips": LPIPS()}

    def forward(self, batch):
        return self.model(batch["rays"])

    def preprocess_data(self, batch, stage):
        if "index" in batch:  # validation / testing
            index = batch["index"]
        else:
            if self.config.model.batch_image_sampling:
                index = torch.randint(
                    0,
                    len(self.dataset.all_images),
                    size=(self.train_num_rays,),
                    device=self.dataset.all_images.device,
                )
            else:
                index = torch.randint(
                    0,
                    len(self.dataset.all_images),
                    size=(1,),
                    device=self.dataset.all_images.device,
                )
            
        if stage in ["train"]:
            c2w = self.dataset.all_c2w[index]
            batch.update({"cam_idx": index})
            x = torch.randint(
                0,
                self.dataset.w,
                size=(self.train_num_rays,),
                device=self.dataset.all_images.device,
            )
            y = torch.randint(
                0,
                self.dataset.h,
                size=(self.train_num_rays,),
                device=self.dataset.all_images.device,
            )
            if self.dataset.directions.ndim == 3:  # (H, W, 3)
                directions = self.dataset.directions[y, x]
            elif self.dataset.directions.ndim == 4:  # (N, H, W, 3)
                directions = self.dataset.directions[index, y, x]
            rays_o, rays_d = get_rays(directions, c2w)
            rgb = (
                self.dataset.all_images[index, y, x]
                .view(-1, self.dataset.all_images.shape[-1])
                .to(self.rank)
            )
        else:
            c2w = self.dataset.all_c2w[index][0]
            img_path = self.dataset.all_paths[index]
            if self.dataset.directions.ndim == 3:  # (H, W, 3)
                directions = self.dataset.directions
            elif self.dataset.directions.ndim == 4:  # (N, H, W, 3)
                directions = self.dataset.directions[index][0]
            rays_o, rays_d = get_rays(directions, c2w)
            rgb = (
                self.dataset.all_images[index]
                .view(-1, self.dataset.all_images.shape[-1])
                .to(self.rank)
            )
            if self.dataset.has_mask:
                fg_mask = self.dataset.all_fg_masks[index].view(-1).to(self.rank)
            batch.update({"img_path": img_path})

        rays = torch.cat([rays_o, F.normalize(rays_d, p=2, dim=-1)], dim=-1)

        if stage in ["train"]:
            if self.config.model.background_color == "white":
                self.model.background_color = torch.ones(
                    (3,), dtype=torch.float32, device=self.rank
                )
            elif self.config.model.background_color == "black":
                self.model.background_color = torch.zeros(
                    (3,), dtype=torch.float32, device=self.rank
                )
            elif self.config.model.background_color == "random":
                self.model.background_color = torch.rand(
                    (3,), dtype=torch.float32, device=self.rank
                )
            elif self.config.model.background_color == "semi-white":
                self.model.background_color = (
                    torch.ones((3,), dtype=torch.float32, device=self.rank)
                    * 206.0
                    / 255.0
                )
            else:
                raise NotImplementedError
        else:
            self.model.background_color = torch.zeros(
                (3,), dtype=torch.float32, device=self.rank
            )

        if self.dataset.apply_mask and stage not in ["train"]:
            rgb = rgb * fg_mask[..., None] + self.model.background_color * (
                1 - fg_mask[..., None]
            )
        if stage in ["train"]:
            # masks are only used for evaluation
            batch.update({"rays": rays, "rgb": rgb})
        else:
            if self.dataset.has_mask:
                batch.update({"rays": rays, "rgb": rgb, "fg_mask": fg_mask})
            else:
                batch.update({"rays": rays, "rgb": rgb})

    def training_step(self, batch, batch_idx):
        out = self(batch)

        loss = 0.0

        # update train_num_rays
        self.log("train/num_samples", float(out["num_samples_full"].sum().item()))
        if self.config.model.dynamic_ray_sampling:
            train_num_rays = int(
                self.train_num_rays
                * (self.train_num_samples / out["num_samples_full"].sum().item())
            )
            self.train_num_rays = min(
                int(self.train_num_rays * 0.9 + train_num_rays * 0.1),
                self.config.model.max_train_num_rays,
            )

        loss_rgb_mse = F.mse_loss(
            out["comp_rgb_full"][out["rays_valid_full"][..., 0]],
            batch["rgb"][out["rays_valid_full"][..., 0]],
        )

        loss_rgb_l1 = F.l1_loss(
            out["comp_rgb_full"][out["rays_valid_full"][..., 0]],
            batch["rgb"][out["rays_valid_full"][..., 0]],
        )

        self.log("train/loss_rgb", loss_rgb_l1)
        loss += loss_rgb_l1 * self.C(self.config.system.loss.lambda_rgb_l1)

        loss_eikonal = (
            (torch.linalg.norm(out["sdf_grad_samples"], ord=2, dim=-1) - 1.0) ** 2
        ).mean()
        self.log("train/loss_eikonal", loss_eikonal)
        loss += loss_eikonal * self.C(self.config.system.loss.lambda_eikonal)

        losses_model_reg = self.model.regularizations(out)
        for name, value in losses_model_reg.items():
            self.log(f"train/loss_{name}", value)
            loss_ = value * self.C(self.config.system.loss[f"lambda_{name}"])
            loss += loss_

        self.log("train/inv_s", out["inv_s"], prog_bar=True)

        for name, value in self.config.system.loss.items():
            if name.startswith("lambda"):
                self.log(f"train_params/{name}", self.C(value))

        self.log("train/num_rays", float(self.train_num_rays), prog_bar=True)

        return {"loss": loss}

    """
    # aggregate outputs from different devices (DP)
    def training_step_end(self, out):
        pass
    """

    """
    # aggregate outputs from different iterations
    def training_epoch_end(self, out):
        pass
    """

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

    """
    # aggregate outputs from different devices when using DP
    def validation_step_end(self, out):
        pass
    """

    def validation_epoch_end(self, out):
        out = self.all_gather(out)
        if self.trainer.is_global_zero:
            out_set = {}
            for step_out in out:
                # DP
                if step_out["index"].ndim == 1:
                    if self.dataset.has_mask:
                        out_set[step_out["index"].item()] = {
                            "psnr": step_out["psnr"],
                            "masked_psnr": step_out["masked_psnr"],
                        }
                    else:
                        out_set[step_out["index"].item()] = {
                            "psnr": step_out["psnr"],
                        }
                # DDP
                else:
                    for oi, index in enumerate(step_out["index"]):
                        if self.dataset.has_mask:
                            out_set[index[0].item()] = {
                                "psnr": step_out["psnr"][oi],
                                "masked_psnr": step_out["masked_psnr"][oi],
                            }
                        else:
                            out_set[index[0].item()] = {
                                "psnr": step_out["psnr"][oi],
                            }
            psnr = torch.mean(torch.stack([o["psnr"] for o in out_set.values()]))
            if self.dataset.has_mask:
                masked_psnr = torch.mean(
                    torch.stack([o["masked_psnr"] for o in out_set.values()])
                )
                self.log(
                    "val/masked_psnr",
                    masked_psnr,
                    prog_bar=True,
                    rank_zero_only=True,
                    sync_dist=True,
                )
            self.log(
                "val/psnr", psnr, prog_bar=True, rank_zero_only=True, sync_dist=True
            )

    def test_step(self, batch, batch_idx):
        out = self(batch)
        W, H = self.dataset.img_wh
        psnr = self.criterions["psnr"](
            out["comp_rgb_full"].to(batch["rgb"]), batch["rgb"]
        )
        ssim = self.criterions["ssim"](
            out["comp_rgb_full"].to(batch["rgb"]).view(H, W, 3),
            batch["rgb"].view(H, W, 3),
        )
        lpips_alex = self.criterions["lpips"](
            out["comp_rgb_full"].to(batch["rgb"]).view(H, W, 3),
            batch["rgb"].view(H, W, 3),
        )
        lpips_vgg = self.criterions["lpips"](
            out["comp_rgb_full"].to(batch["rgb"]).view(H, W, 3),
            batch["rgb"].view(H, W, 3),
            False,
        )
        average_alex = compute_avg_error(psnr, ssim, lpips_alex)
        average_vgg = compute_avg_error(psnr, ssim, lpips_vgg)

        result = {
            "img_path": batch["img_path"],
            "psnr": psnr,
            "ssim": ssim,
            "lpips_alex": lpips_alex,
            "lpips_vgg": lpips_vgg,
            "average_alex": average_alex,
            "average_vgg": average_vgg,
            "index": batch["index"],
        }

        if self.dataset.has_mask:
            masked_psnr = self.criterions["psnr"](
                out["comp_rgb_full"].to(batch["rgb"]),
                batch["rgb"],
                valid_mask=batch["fg_mask"].bool(),
            )
            masked_rgb_out = out["comp_rgb_full"].to(batch["rgb"]) * batch["fg_mask"][
                ..., None
            ] + self.model.background_color * (1 - batch["fg_mask"][..., None])
            masked_gt = batch["rgb"] * batch["fg_mask"][
                ..., None
            ] + self.model.background_color * (1 - batch["fg_mask"][..., None])
            masked_ssim = self.criterions["ssim"](
                masked_rgb_out.view(H, W, 3), masked_gt.view(H, W, 3)
            )
            masked_lpips_alex = self.criterions["lpips"](
                masked_rgb_out.view(H, W, 3), masked_gt.view(H, W, 3)
            )
            masked_lpips_vgg = self.criterions["lpips"](
                masked_rgb_out.view(H, W, 3), masked_gt.view(H, W, 3), False
            )
            masked_average_alex = compute_avg_error(
                masked_psnr, masked_ssim, masked_lpips_alex
            )
            masked_average_vgg = compute_avg_error(
                masked_psnr, masked_ssim, masked_lpips_vgg
            )
            result = {
                "img_path": batch["img_path"],
                "psnr": psnr,
                "ssim": ssim,
                "lpips_alex": lpips_alex,
                "lpips_vgg": lpips_vgg,
                "average_alex": average_alex,
                "average_vgg": average_vgg,
                "masked_average_alex": masked_average_alex,
                "masked_average_vgg": masked_average_vgg,
                "masked_psnr": masked_psnr,
                "masked_ssim": masked_ssim,
                "masked_lpips_alex": masked_lpips_alex,
                "masked_lpips_vgg": masked_lpips_vgg,
                "index": batch["index"],
            }
        self.save_image_grid(
            f"it{self.global_step}-test/{batch['index'][0].item()}.png",
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
            + (
                [
                    {
                        "type": "rgb",
                        "img": masked_rgb_out.view(H, W, 3),
                        "kwargs": {"data_format": "HWC"},
                    },
                    {
                        "type": "rgb",
                        "img": masked_gt.view(H, W, 3),
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if self.dataset.has_mask
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
        return result

    def test_epoch_end(self, out):
        """
        Synchronize devices.
        Generate image sequence using test outputs.
        """
        out = self.all_gather(out)
        if self.trainer.is_global_zero:
            out_set = {}
            for step_out in out:
                # DP
                if step_out["index"].ndim == 1:
                    if self.dataset.has_mask:
                        out_set[step_out["index"].item()] = {
                            "psnr": step_out["psnr"],
                            "masked_psnr": step_out["masked_psnr"],
                            "ssim": step_out["ssim"],
                            "masked_ssim": step_out["masked_ssim"],
                            "lpips_alex": step_out["lpips_alex"],
                            "masked_lpips_alex": step_out["masked_lpips_alex"],
                            "lpips_vgg": step_out["lpips_vgg"],
                            "masked_lpips_vgg": step_out["masked_lpips_vgg"],
                            "average_alex": step_out["average_alex"],
                            "average_vgg": step_out["average_vgg"],
                            "masked_average_alex": step_out["masked_average_alex"],
                            "masked_average_vgg": step_out["masked_average_vgg"],
                            "img_path": step_out["img_path"],
                        }
                    else:
                        out_set[step_out["index"].item()] = {
                            "psnr": step_out["psnr"],
                            "ssim": step_out["ssim"],
                            "lpips_alex": step_out["lpips_alex"],
                            "lpips_vgg": step_out["lpips_vgg"],
                            "average_alex": step_out["average_alex"],
                            "average_vgg": step_out["average_vgg"],
                            "img_path": step_out["img_path"],
                        }
                # DDP
                else:
                    for oi, index in enumerate(step_out["index"]):
                        if self.dataset.has_mask:
                            out_set[index[0].item()] = {
                                "psnr": step_out["psnr"][oi],
                                "masked_psnr": step_out["masked_psnr"][oi],
                                "ssim": step_out["ssim"][oi],
                                "masked_ssim": step_out["masked_ssim"][oi],
                                "lpips_alex": step_out["lpips_alex"][oi],
                                "masked_lpips_alex": step_out["masked_lpips_alex"][oi],
                                "lpips_vgg": step_out["lpips_vgg"][oi],
                                "masked_lpips_vgg": step_out["masked_lpips_vgg"][oi],
                                "average_alex": step_out["average_alex"][oi],
                                "average_vgg": step_out["average_vgg"][oi],
                                "masked_average_alex": step_out["masked_average_alex"][
                                    oi
                                ],
                                "masked_average_vgg": step_out["masked_average_vgg"][
                                    oi
                                ],
                                "img_path": step_out["img_path"],
                            }
                        else:
                            out_set[index[0].item()] = {
                                "psnr": step_out["psnr"][oi],
                                "ssim": step_out["ssim"][oi],
                                "lpips_alex": step_out["lpips_alex"][oi],
                                "lpips_vgg": step_out["lpips_vgg"][oi],
                                "average_alex": step_out["average_alex"][oi],
                                "average_vgg": step_out["average_vgg"][oi],
                                "img_path": step_out["img_path"],
                            }
            # export dict with results
            if self.dataset.has_mask:
                per_img_result_dict = [
                    {
                        "image_name": o["img_path"],
                        "psnr": o["psnr"].cpu().numpy().item(),
                        "masked_psnr": o["masked_psnr"].cpu().numpy().item(),
                        "ssim": o["ssim"].cpu().numpy().item(),
                        "masked_ssim": o["masked_ssim"].cpu().numpy().item(),
                        "lpips_alex": o["lpips_alex"].cpu().numpy().item(),
                        "masked_lpips_alex": o["masked_lpips_alex"]
                        .cpu()
                        .numpy()
                        .item(),
                        "lpips_vgg": o["lpips_vgg"].cpu().numpy().item(),
                        "masked_lpips_vgg": o["masked_lpips_vgg"].cpu().numpy().item(),
                        "average_alex": o["average_alex"].cpu().numpy().item(),
                        "average_vgg": o["average_vgg"].cpu().numpy().item(),
                        "masked_average_alex": o["masked_average_alex"]
                        .cpu()
                        .numpy()
                        .item(),
                        "masked_average_vgg": o["masked_average_vgg"]
                        .cpu()
                        .numpy()
                        .item(),
                    }
                    for o in out_set.values()
                ]
            else:
                per_img_result_dict = [
                    {
                        "image_name": o["img_path"],
                        "psnr": o["psnr"].cpu().numpy().item(),
                        "ssim": o["ssim"].cpu().numpy().item(),
                        "lpips_alex": o["lpips_alex"].cpu().numpy().item(),
                        "lpips_vgg": o["lpips_vgg"].cpu().numpy().item(),
                        "average_alex": o["average_alex"].cpu().numpy().item(),
                        "average_vgg": o["average_vgg"].cpu().numpy().item(),
                    }
                    for o in out_set.values()
                ]
            self.save_json(
                f"it{self.global_step}-test-perimage-results.json", per_img_result_dict
            )
            if self.dataset.has_mask:
                masked_psnr = torch.mean(
                    torch.stack([o["masked_psnr"] for o in out_set.values()])
                )
                masked_ssim = torch.mean(
                    torch.stack([o["masked_ssim"] for o in out_set.values()])
                )
                masked_lpips_alex = torch.mean(
                    torch.stack([o["masked_lpips_alex"] for o in out_set.values()])
                )
                masked_lpips_vgg = torch.mean(
                    torch.stack([o["masked_lpips_vgg"] for o in out_set.values()])
                )
                masked_average_alex = torch.mean(
                    torch.stack([o["masked_average_alex"] for o in out_set.values()])
                )
                masked_average_vgg = torch.mean(
                    torch.stack([o["masked_average_vgg"] for o in out_set.values()])
                )
            psnr = torch.mean(torch.stack([o["psnr"] for o in out_set.values()]))

            ssim = torch.mean(torch.stack([o["ssim"] for o in out_set.values()]))

            lpips_alex = torch.mean(
                torch.stack([o["lpips_alex"] for o in out_set.values()])
            )

            lpips_vgg = torch.mean(
                torch.stack([o["lpips_vgg"] for o in out_set.values()])
            )

            average_alex = torch.mean(
                torch.stack([o["average_alex"] for o in out_set.values()])
            )

            average_vgg = torch.mean(
                torch.stack([o["average_vgg"] for o in out_set.values()])
            )
            if self.dataset.has_mask:
                average_result_dict = [
                    {
                        "psnr": psnr.cpu().numpy().item(),
                        "masked_psnr": masked_psnr.cpu().numpy().item(),
                        "ssim": ssim.cpu().numpy().item(),
                        "masked_ssim": masked_ssim.cpu().numpy().item(),
                        "lpips_alex": lpips_alex.cpu().numpy().item(),
                        "masked_lpips_alex": masked_lpips_alex.cpu().numpy().item(),
                        "lpips_vgg": lpips_vgg.cpu().numpy().item(),
                        "masked_lpips_vgg": masked_lpips_vgg.cpu().numpy().item(),
                        "average_alex": average_alex.cpu().numpy().item(),
                        "masked_average_alex": masked_average_alex.cpu().numpy().item(),
                        "average_vgg": average_vgg.cpu().numpy().item(),
                        "masked_average_vgg": masked_average_vgg.cpu().numpy().item(),
                    }
                ]
            else:
                average_result_dict = [
                    {
                        "psnr": psnr.cpu().numpy().item(),
                        "ssim": ssim.cpu().numpy().item(),
                        "lpips_alex": lpips_alex.cpu().numpy().item(),
                        "lpips_vgg": lpips_vgg.cpu().numpy().item(),
                        "average_alex": average_alex.cpu().numpy().item(),
                        "average_vgg": average_vgg.cpu().numpy().item(),
                    }
                ]
            self.save_json(
                f"it{self.global_step}-test-average-results.json", average_result_dict
            )
            if self.dataset.has_mask:
                self.log(
                    "test/masked_psnr",
                    masked_psnr,
                    prog_bar=True,
                    rank_zero_only=True,
                    sync_dist=True,
                )
                self.log(
                    "test/masked_ssim",
                    masked_ssim,
                    prog_bar=True,
                    rank_zero_only=True,
                    sync_dist=True,
                )
                self.log(
                    "test/masked_lpips_alex",
                    masked_lpips_alex,
                    prog_bar=True,
                    rank_zero_only=True,
                    sync_dist=True,
                )
                self.log(
                    "test/masked_lpips_vgg",
                    masked_lpips_vgg,
                    prog_bar=True,
                    rank_zero_only=True,
                    sync_dist=True,
                )
                self.log(
                    "test/masked_average_alex",
                    masked_average_alex,
                    prog_bar=True,
                    rank_zero_only=True,
                    sync_dist=True,
                )
                self.log(
                    "test/masked_average_vgg",
                    masked_average_vgg,
                    prog_bar=True,
                    rank_zero_only=True,
                    sync_dist=True,
                )
            self.log(
                "test/psnr", psnr, prog_bar=True, rank_zero_only=True, sync_dist=True
            )

            self.log(
                "test/ssim", ssim, prog_bar=True, rank_zero_only=True, sync_dist=True
            )

            self.log(
                "test/lpips_alex",
                lpips_alex,
                prog_bar=True,
                rank_zero_only=True,
                sync_dist=True,
            )

            self.log(
                "test/lpips_vgg",
                lpips_vgg,
                prog_bar=True,
                rank_zero_only=True,
                sync_dist=True,
            )

            self.log(
                "test/average_alex",
                average_alex,
                prog_bar=True,
                rank_zero_only=True,
                sync_dist=True,
            )

            self.log(
                "test/average_vgg",
                average_vgg,
                prog_bar=True,
                rank_zero_only=True,
                sync_dist=True,
            )

            self.export()

    def export(self):
        if self.config.export.get("export_mesh", False):
            mesh = self.model.export(self.config.export)
            self.save_mesh(
                f"it{self.global_step}-{self.config.model.geometry.isosurface.method}{self.config.model.geometry.isosurface.resolution}.ply",
                **mesh,
            )

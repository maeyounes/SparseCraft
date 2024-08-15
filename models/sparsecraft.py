import models
from models.neus import NeuSModel
import torch
import torch.nn.functional as F
from nerfacc import (
    ray_marching,
    render_weight_from_alpha,
    accumulate_along_rays,
)
from nerfacc import (
    ContractionType,
    OccupancyGrid,
)
import math
from models.utils import get_activation
from models.utils import chunk_batch
from systems.utils import update_module_step
from models.neus import VarianceNetwork

@models.register("sparsecraft")
class SparseCraftModel(NeuSModel):
    def setup(self):
        self.geometry = models.make(self.config.geometry.name, self.config.geometry)
        self.texture = models.make(self.config.texture.name, self.config.texture)
        self.geometry.contraction_type = ContractionType.AABB
        self.occupancy_grid_res = self.config.get("occupancy_grid_res", 128)
        self.occupancy_grid_res_bg = self.config.get("occupancy_grid_res_bg", 256)

        if self.config.learned_background:
            self.geometry_bg = models.make(
                self.config.geometry_bg.name, self.config.geometry_bg
            )
            self.texture_bg = models.make(
                self.config.texture_bg.name, self.config.texture_bg
            )
            self.geometry_bg.contraction_type = ContractionType.UN_BOUNDED_SPHERE
            self.near_plane_bg, self.far_plane_bg = (
                self.config.near_plane_bg,
                self.config.far_plane_bg,
            )  # defaults are 0.1, 10 #1e3
            self.cone_angle_bg = (
                10
                ** (math.log10(self.far_plane_bg) / self.config.num_samples_per_ray_bg)
                - 1.0
            )
            self.render_step_size_bg = 0.01

        self.variance = VarianceNetwork(self.config.variance)
        self.register_buffer(
            "scene_aabb",
            torch.as_tensor(
                [
                    -self.config.radius,
                    -self.config.radius,
                    -self.config.radius,
                    self.config.radius,
                    self.config.radius,
                    self.config.radius,
                ],
                dtype=torch.float32,
            ),
        )
        if self.config.grid_prune:
            self.occupancy_grid = OccupancyGrid(
                roi_aabb=self.scene_aabb,
                resolution=self.occupancy_grid_res,
                contraction_type=ContractionType.AABB,
            )
            if self.config.learned_background:
                self.occupancy_grid_bg = OccupancyGrid(
                    roi_aabb=self.scene_aabb,
                    resolution=self.occupancy_grid_res_bg,
                    contraction_type=ContractionType.UN_BOUNDED_SPHERE,
                )
        self.randomized = self.config.randomized
        self.background_color = None
        self.render_step_size = (
            1.732 * 2 * self.config.radius / self.config.num_samples_per_ray
        )

    def update_step(self, epoch, global_step):
        update_module_step(self.geometry, epoch, global_step)
        update_module_step(self.texture, epoch, global_step)
        if self.config.learned_background:
            update_module_step(self.geometry_bg, epoch, global_step)
            update_module_step(self.texture_bg, epoch, global_step)
        update_module_step(self.variance, epoch, global_step)

        cos_anneal_end = self.config.get("cos_anneal_end", 0)
        self.cos_anneal_ratio = (
            1.0 if cos_anneal_end == 0 else min(1.0, global_step / cos_anneal_end)
        )

        def occ_eval_fn(x):
            sdf = self.geometry(x, with_grad=False, with_feature=False)
            inv_s = self.variance(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)
            inv_s = inv_s.expand(sdf.shape[0], 1)
            cdf = torch.sigmoid(torch.unsqueeze(sdf, -1) * inv_s)
            e = inv_s * (1 - cdf) * self.render_step_size
            alpha = (1 - torch.exp(-e)).view(-1).clip(0.0, 1.0)
            return alpha

        def occ_eval_fn_bg(x):
            density, _ = self.geometry_bg(x)
            # approximate for 1 - torch.exp(-density[...,None] * self.render_step_size_bg) based on taylor series
            return density[..., None] * self.render_step_size_bg

        if self.training and self.config.grid_prune:
            self.occupancy_grid.every_n_step(
                step=global_step,
                occ_eval_fn=occ_eval_fn,
                occ_thre=self.config.get("grid_prune_occ_thre", 0.01),
            )
            if self.config.learned_background:
                self.occupancy_grid_bg.every_n_step(
                    step=global_step,
                    occ_eval_fn=occ_eval_fn_bg,
                    occ_thre=self.config.get("grid_prune_occ_thre_bg", 0.01),
                )

    def get_alpha(self, sdf, normal, dirs, dists):
        inv_s = self.variance(torch.zeros([1, 3]))[:, :1].clip(
            1e-6, 1e6
        )  # Single parameter
        inv_s = inv_s.expand(sdf.shape[0], 1)
        true_cos = (dirs * normal).sum(-1, keepdim=True)
        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(
            F.relu(-true_cos * 0.5 + 0.5) * (1.0 - self.cos_anneal_ratio)
            + F.relu(-true_cos) * self.cos_anneal_ratio
        )  # always non-positive
        cdf = torch.sigmoid(torch.unsqueeze(sdf, -1) * inv_s)
        e = inv_s * (1 - cdf) * (-iter_cos) * self.render_step_size
        alpha = (1 - torch.exp(-e)).view(-1).clip(0.0, 1.0)
        return alpha, inv_s

    def forward(self, rays, query_points=None, input_points=None):
        if self.training:
            out = self.forward_(rays)
        else:
            out = chunk_batch(self.forward_, self.config.ray_chunk, True, rays)
        return {**out, "inv_s": self.variance.inv_s}
        # return {**out}

    def forward_(self, rays):
        n_rays = rays.shape[0]
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]  # both (N_rays, 3)

        with torch.no_grad():
            ray_indices, t_starts, t_ends = ray_marching(
                rays_o,
                rays_d,
                scene_aabb=self.scene_aabb,
                grid=self.occupancy_grid if self.config.grid_prune else None,
                alpha_fn=None,
                near_plane=None,
                far_plane=None,
                render_step_size=self.render_step_size,
                stratified=self.randomized,
                cone_angle=0.0,
                alpha_thre=0.0,
            )

        ray_indices = ray_indices.long()
        t_origins = rays_o[ray_indices]
        t_dirs = rays_d[ray_indices]
        midpoints = (t_starts + t_ends) / 2.0
        positions = t_origins + t_dirs * midpoints
        dists = t_ends - t_starts

        # additional points to apply eikonal regularization in the whole space instead of only on the surface
        random_pts_nb = 50000
        space_points = (
            (torch.rand((random_pts_nb, 3), device=self.rank) - 0.5)
            * 2
            * self.config.radius
        )
        positions = torch.cat([positions, space_points])
        if self.config.geometry.grad_type == "finite_difference":
            sdf, sdf_grad_all, feature_all, sdf_laplace, encoding_all = self.geometry(
                positions,
                with_grad=True,
                with_feature=True,
                with_laplace=True,
                with_encoding=True,
            )
            sdf_laplace = sdf_laplace[:-random_pts_nb]
        else:
            sdf, sdf_grad_all, feature_all, encoding_all = self.geometry(
                positions,
                with_grad=True,
                with_feature=True,
                with_encoding=True,
            )
        sdf, feature, sdf_grad, positions, encoding = (
            sdf[:-random_pts_nb],
            feature_all[:-random_pts_nb],
            sdf_grad_all[:-random_pts_nb],
            positions[:-random_pts_nb],
            encoding_all[:-random_pts_nb],
        )
        normal = F.normalize(sdf_grad, p=2, dim=-1)
        alphas, inv_s = self.get_alpha(sdf, normal, t_dirs, dists)
        alpha = alphas[..., None]
        albedo_color = self.texture.output_albedo(encoding, feature)
        specular_color = self.texture.output_specular(encoding, feature, t_dirs, normal)
        rgb = albedo_color + specular_color
        rgb = get_activation(self.texture.config.color_activation)(rgb)
        weights = render_weight_from_alpha(
            alpha, ray_indices=ray_indices, n_rays=n_rays
        )
        opacity = accumulate_along_rays(
            weights, ray_indices, values=None, n_rays=n_rays
        )
        depth = accumulate_along_rays(
            weights, ray_indices, values=midpoints, n_rays=n_rays
        )
        comp_rgb = accumulate_along_rays(
            weights, ray_indices, values=rgb, n_rays=n_rays
        )
        comp_normal = accumulate_along_rays(
            weights, ray_indices, values=normal, n_rays=n_rays
        )
        comp_normal = F.normalize(comp_normal, p=2, dim=-1)
        out = {
            "comp_rgb": comp_rgb,
            "comp_normal": comp_normal,
            "opacity": opacity,
            "depth": depth,
            "rays_valid": opacity > 0,
            "num_samples": torch.as_tensor(
                [len(t_starts)], dtype=torch.int32, device=rays.device
            ),
        }

        if self.training:
            out.update(
                {
                    "sdf_samples": sdf,
                    "specular_color": specular_color,
                    "sdf_grad_samples": sdf_grad_all,
                    "weights": weights.view(-1),
                    "alpha": alpha.view(-1),
                    "points": midpoints.view(-1),
                    "intervals": dists.view(-1),
                    "ray_indices": ray_indices.view(-1),
                    "inv_s": inv_s.mean(),
                }
            )
            if self.config.geometry.grad_type == "finite_difference":
                out.update(
                    {
                        "sdf_laplace_samples": sdf_laplace,
                    }
                )
        else:
            comp_normal = accumulate_along_rays(weights, ray_indices, values=normal, n_rays=n_rays)
            comp_normal = F.normalize(comp_normal, p=2, dim=-1)

            depth = accumulate_along_rays(
                weights, ray_indices, values=midpoints, n_rays=n_rays
            )

            out.update({"comp_normal": comp_normal,
                        "depth": depth,
            })


        if self.config.learned_background:
            out_bg = self.forward_bg_(rays)
        else:
            out_bg = {
                "comp_rgb": self.background_color[None, :].expand(*comp_rgb.shape),
                "num_samples": torch.zeros_like(out["num_samples"]),
                "rays_valid": torch.zeros_like(out["rays_valid"]),
            }
        out_full = {
            "comp_rgb": out["comp_rgb"] + out_bg["comp_rgb"] * (1.0 - out["opacity"])
            if (self.config.learned_background or self.config.white_background)
            else out["comp_rgb"],
            "depth": None if self.training else out["depth"],
            "num_samples": out["num_samples"] + out_bg["num_samples"],
            "rays_valid": out["rays_valid"] | out_bg["rays_valid"],
        }

        return {
            **out,
            **{k + "_bg": v for k, v in out_bg.items()},
            **{k + "_full": v for k, v in out_full.items()},
        }

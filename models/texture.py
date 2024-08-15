import torch
import torch.nn as nn

import models
from models.utils import get_activation
from models.network_utils import get_encoding, get_mlp
from systems.utils import update_module_step


@models.register("volume-radiance")
class VolumeRadiance(nn.Module):
    def __init__(self, config):
        super(VolumeRadiance, self).__init__()
        self.config = config
        self.n_dir_dims = self.config.get("n_dir_dims", 3)
        self.n_output_dims = 3
        encoding = get_encoding(self.n_dir_dims, self.config.dir_encoding_config)
        self.n_input_dims = self.config.input_feature_dim + encoding.n_output_dims
        network = get_mlp(
            self.n_input_dims, self.n_output_dims, self.config.mlp_network_config
        )
        self.encoding = encoding
        self.network = network

    def forward(self, features, dirs, *args):
        dirs = (dirs + 1.0) / 2.0  # (-1, 1) => (0, 1)
        dirs_embd = self.encoding(dirs.view(-1, self.n_dir_dims))
        network_inp = torch.cat(
            [features.view(-1, features.shape[-1]), dirs_embd]
            + [arg.view(-1, arg.shape[-1]) for arg in args],
            dim=-1,
        )
        color = (
            self.network(network_inp)
            .view(*features.shape[:-1], self.n_output_dims)
            .float()
        )
        if "color_activation" in self.config:
            color = get_activation(self.config.color_activation)(color)
        return color

    def update_step(self, epoch, global_step):
        update_module_step(self.encoding, epoch, global_step)

    def regularizations(self, out):
        return {}


@models.register("disentangled-volume-radiance")
class DisentangledVolumeRadiance(nn.Module):
    def __init__(self, config):
        super(DisentangledVolumeRadiance, self).__init__()
        self.config = config
        self.n_dir_dims = self.config.get("n_dir_dims", 3)
        self.albedo_n_output_dims = 3
        self.specular_n_output_dims = 1
        encoding = get_encoding(self.n_dir_dims, self.config.dir_encoding_config)
        self.albedo_n_input_dims = (
            self.config.input_feature_dim - 3 + 67
        )  # point locations and features from the SDF MLP as input
        self.specular_n_input_dims = (
            self.config.input_feature_dim + encoding.n_output_dims + 67
        )  # point locations, SDF MLP features, reflection direction and view direction
        albedo_network = get_mlp(
            self.albedo_n_input_dims,
            self.albedo_n_output_dims,
            self.config.mlp_network_config,
        )
        specular_network = get_mlp(
            self.specular_n_input_dims,
            self.specular_n_output_dims,
            self.config.mlp_network_config,
        )
        self.encoding = encoding
        self.albedo_network = albedo_network
        self.specular_network = specular_network

    def forward(self, input_enc, features, dirs, *args):
        dirs = (dirs + 1.0) / 2.0  # (-1, 1) => (0, 1)
        dirs_embd = self.encoding(dirs.view(-1, self.n_dir_dims))
        albedo_network_inp = torch.cat(
            [input_enc, features.view(-1, features.shape[-1])],
            dim=-1,
        )
        specular_network_inp = torch.cat(
            [input_enc, features.view(-1, features.shape[-1]), dirs_embd]
            + [arg.view(-1, arg.shape[-1]) for arg in args],
            dim=-1,
        )
        albedo_color = (
            self.albedo_network(albedo_network_inp)
            .view(*features.shape[:-1], self.albedo_n_output_dims)
            .float()
        )
        specular_color = (
            self.specular_network(specular_network_inp)
            .view(*features.shape[:-1], self.specular_n_output_dims)
            .float()
        )
        color = albedo_color + specular_color
        if "color_activation" in self.config:
            color = get_activation(self.config.color_activation)(color)
        return color

    def forward_albedo(self, input_enc, features):
        albedo_network_inp = torch.cat(
            [input_enc, features.view(-1, features.shape[-1])],
            dim=-1,
        )
        albedo_color = (
            self.albedo_network(albedo_network_inp)
            .view(*features.shape[:-1], self.albedo_n_output_dims)
            .float()
        )
        if "color_activation" in self.config:
            color = get_activation(self.config.color_activation)(albedo_color)
        return color

    def forward_specular(self, input_enc, features, dirs, *args):
        dirs = (dirs + 1.0) / 2.0  # (-1, 1) => (0, 1)
        dirs_embd = self.encoding(dirs.view(-1, self.n_dir_dims))
        specular_network_inp = torch.cat(
            [input_enc, features.view(-1, features.shape[-1]), dirs_embd]
            + [arg.view(-1, arg.shape[-1]) for arg in args],
            dim=-1,
        )
        specular_color = (
            self.specular_network(specular_network_inp)
            .view(*features.shape[:-1], self.specular_n_output_dims)
            .float()
        )
        if "color_activation" in self.config:
            color = get_activation(self.config.color_activation)(specular_color)
        return color

    def output_albedo(self, input_enc, features):
        albedo_network_inp = torch.cat(
            [input_enc, features.view(-1, features.shape[-1])],
            dim=-1,
        )
        albedo_color = (
            self.albedo_network(albedo_network_inp)
            .view(*features.shape[:-1], self.albedo_n_output_dims)
            .float()
        )
        return albedo_color

    def output_specular(self, input_enc, features, dirs, *args):
        dirs = (dirs + 1.0) / 2.0  # (-1, 1) => (0, 1)
        dirs_embd = self.encoding(dirs.view(-1, self.n_dir_dims))
        specular_network_inp = torch.cat(
            [input_enc, features.view(-1, features.shape[-1]), dirs_embd]
            + [arg.view(-1, arg.shape[-1]) for arg in args],
            dim=-1,
        )
        specular_color = (
            self.specular_network(specular_network_inp)
            .view(*features.shape[:-1], self.specular_n_output_dims)
            .float()
        )
        return specular_color

    def update_step(self, epoch, global_step):
        update_module_step(self.encoding, epoch, global_step)

    def regularizations(self, out):
        return {}
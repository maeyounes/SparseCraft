import torch
import torch.nn as nn
import torch.nn.functional as F
import skimage
import lpips
import numpy as np


class WeightedLoss(nn.Module):
    @property
    def func(self):
        raise NotImplementedError

    def forward(self, inputs, targets, weight=None, reduction="mean"):
        assert reduction in ["none", "sum", "mean", "valid_mean"]
        loss = self.func(inputs, targets, reduction="none")
        if weight is not None:
            while weight.ndim < inputs.ndim:
                weight = weight[..., None]
            loss *= weight.float()
        if reduction == "none":
            return loss
        elif reduction == "sum":
            return loss.sum()
        elif reduction == "mean":
            return loss.mean()
        elif reduction == "valid_mean":
            return loss.sum() / weight.float().sum()


class MSELoss(WeightedLoss):
    @property
    def func(self):
        return F.mse_loss


class L1Loss(WeightedLoss):
    @property
    def func(self):
        return F.l1_loss


class PSNR(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, inputs, targets, valid_mask=None, regnerf_mask=None, reduction="mean"
    ):
        assert reduction in ["mean", "none"]
        if valid_mask is not None:
            value = ((inputs - targets) ** 2)[valid_mask]
        elif regnerf_mask is not None:
            value = ((inputs - targets)[regnerf_mask]) ** 2
        else:
            value = (inputs - targets) ** 2
        if reduction == "mean":
            return -10 * torch.log10(torch.mean(value))
        elif reduction == "none":
            return -10 * torch.log10(
                torch.mean(value, dim=tuple(range(value.ndim)[1:]))
            )


def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


class SSIM(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        inputs = normalize_to_neg_one_to_one(inputs).cpu().numpy()
        targets = normalize_to_neg_one_to_one(targets).cpu().numpy()
        return float(
            skimage.metrics.structural_similarity(inputs, targets, channel_axis=-1)
        )


def compute_avg_error(psnr, ssim, lpips):
    """The 'average' error used in regnerf."""
    mse = psnr_to_mse(psnr)
    dssim = np.sqrt(1 - ssim)
    return np.exp(np.mean(np.log(np.array([mse, dssim, lpips]))))


def psnr_to_mse(psnr):
    """Compute MSE given a PSNR (we assume the maximum pixel value is 1)."""
    """Adapted from RegNerf"""
    return np.exp(-0.1 * np.log(10.0) * float(psnr))


class LPIPS(nn.Module):
    def __init__(self):
        super().__init__()
        self.lpips_alex_loss = lpips.LPIPS(net="alex").cuda()
        self.lpips_vgg_loss = lpips.LPIPS(net="vgg").cuda()

    def forward(self, inputs, targets, alex=True):
        inputs = normalize_to_neg_one_to_one(inputs)
        targets = normalize_to_neg_one_to_one(targets)
        inputs = torch.moveaxis(inputs, -1, 0)
        targets = torch.moveaxis(targets, -1, 0)
        if alex:
            return float(self.lpips_alex_loss(inputs, targets).squeeze())
        else:
            return float(self.lpips_vgg_loss(inputs, targets).squeeze())


class SSIM_0:
    def __init__(
        self,
        data_range=(0, 1),
        kernel_size=(11, 11),
        sigma=(1.5, 1.5),
        k1=0.01,
        k2=0.03,
        gaussian=True,
    ):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.gaussian = gaussian

        if any(x % 2 == 0 or x <= 0 for x in self.kernel_size):
            raise ValueError(
                f"Expected kernel_size to have odd positive number. Got {kernel_size}."
            )
        if any(y <= 0 for y in self.sigma):
            raise ValueError(f"Expected sigma to have positive number. Got {sigma}.")

        data_scale = data_range[1] - data_range[0]
        self.c1 = (k1 * data_scale) ** 2
        self.c2 = (k2 * data_scale) ** 2
        self.pad_h = (self.kernel_size[0] - 1) // 2
        self.pad_w = (self.kernel_size[1] - 1) // 2
        self._kernel = self._gaussian_or_uniform_kernel(
            kernel_size=self.kernel_size, sigma=self.sigma
        )

    def _uniform(self, kernel_size):
        max, min = 2.5, -2.5
        ksize_half = (kernel_size - 1) * 0.5
        kernel = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
        for i, j in enumerate(kernel):
            if min <= j <= max:
                kernel[i] = 1 / (max - min)
            else:
                kernel[i] = 0

        return kernel.unsqueeze(dim=0)  # (1, kernel_size)

    def _gaussian(self, kernel_size, sigma):
        ksize_half = (kernel_size - 1) * 0.5
        kernel = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
        gauss = torch.exp(-0.5 * (kernel / sigma).pow(2))
        return (gauss / gauss.sum()).unsqueeze(dim=0)  # (1, kernel_size)

    def _gaussian_or_uniform_kernel(self, kernel_size, sigma):
        if self.gaussian:
            kernel_x = self._gaussian(kernel_size[0], sigma[0])
            kernel_y = self._gaussian(kernel_size[1], sigma[1])
        else:
            kernel_x = self._uniform(kernel_size[0])
            kernel_y = self._uniform(kernel_size[1])

        return torch.matmul(
            kernel_x.t(), kernel_y
        )  # (kernel_size, 1) * (1, kernel_size)

    def __call__(self, output, target, reduction="mean"):
        if output.dtype != target.dtype:
            raise TypeError(
                f"Expected output and target to have the same data type. Got output: {output.dtype} and y: {target.dtype}."
            )

        if output.shape != target.shape:
            raise ValueError(
                f"Expected output and target to have the same shape. Got output: {output.shape} and y: {target.shape}."
            )

        if len(output.shape) != 4 or len(target.shape) != 4:
            raise ValueError(
                f"Expected output and target to have BxCxHxW shape. Got output: {output.shape} and y: {target.shape}."
            )

        assert reduction in ["mean", "sum", "none"]

        channel = output.size(1)
        if len(self._kernel.shape) < 4:
            self._kernel = self._kernel.expand(channel, 1, -1, -1)

        output = F.pad(
            output, [self.pad_w, self.pad_w, self.pad_h, self.pad_h], mode="reflect"
        )
        target = F.pad(
            target, [self.pad_w, self.pad_w, self.pad_h, self.pad_h], mode="reflect"
        )

        input_list = torch.cat(
            [output, target, output * output, target * target, output * target]
        )
        outputs = F.conv2d(input_list, self._kernel, groups=channel)

        output_list = [
            outputs[x * output.size(0) : (x + 1) * output.size(0)]
            for x in range(len(outputs))
        ]

        mu_pred_sq = output_list[0].pow(2)
        mu_target_sq = output_list[1].pow(2)
        mu_pred_target = output_list[0] * output_list[1]

        sigma_pred_sq = output_list[2] - mu_pred_sq
        sigma_target_sq = output_list[3] - mu_target_sq
        sigma_pred_target = output_list[4] - mu_pred_target

        a1 = 2 * mu_pred_target + self.c1
        a2 = 2 * sigma_pred_target + self.c2
        b1 = mu_pred_sq + mu_target_sq + self.c1
        b2 = sigma_pred_sq + sigma_target_sq + self.c2

        ssim_idx = (a1 * a2) / (b1 * b2)
        _ssim = torch.mean(ssim_idx, (1, 2, 3))

        if reduction == "none":
            return _ssim
        elif reduction == "sum":
            return _ssim.sum()
        elif reduction == "mean":
            return _ssim.mean()


def binary_cross_entropy(input, target):
    """
    F.binary_cross_entropy is not numerically stable in mixed-precision training.
    """
    return -(target * torch.log(input) + (1 - target) * torch.log(1 - input)).mean()

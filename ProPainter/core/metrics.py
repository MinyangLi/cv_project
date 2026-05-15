# -*- coding: utf-8 -*-
"""
Image / video evaluation metrics for ProPainter.

Includes: PSNR, SSIM, MSE, MAE, LPIPS (optional), VFID (I3D Frechet distance).
"""
import numpy as np
from scipy import linalg

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.utils import to_tensors
import lpips
# ---------------------------------------------------------------------------
# Pixel-space metrics (inputs: uint8 or float, same shape, typically [0, 255])
# ---------------------------------------------------------------------------


def calculate_mse(img1, img2):
    """Mean squared error in pixel space (same scale as inputs)."""
    img1 = np.asarray(img1, dtype=np.float64)
    img2 = np.asarray(img2, dtype=np.float64)
    assert img1.shape == img2.shape, (
        f'Image shapes differ: {img1.shape}, {img2.shape}.')
    return float(np.mean((img1 - img2) ** 2))


def calculate_mae(img1, img2):
    """Mean absolute error in pixel space."""
    img1 = np.asarray(img1, dtype=np.float64)
    img2 = np.asarray(img2, dtype=np.float64)
    assert img1.shape == img2.shape, (
        f'Image shapes differ: {img1.shape}, {img2.shape}.')
    return float(np.mean(np.abs(img1 - img2)))


def calculate_psnr(img1, img2, data_range=255.0):
    """PSNR (Peak Signal-to-Noise Ratio). Default assumes [0, 255] uint8 images."""
    img1 = np.asarray(img1, dtype=np.float64)
    img2 = np.asarray(img2, dtype=np.float64)
    assert img1.shape == img2.shape, (
        f'Image shapes differ: {img1.shape}, {img2.shape}.')
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return float(20.0 * np.log10(data_range / np.sqrt(mse)))


def _adaptive_ssim_win_size(h, w, max_win=65):
    """Odd win_size in [3, max_win], not larger than min spatial side."""
    side = min(h, w)
    win_size = min(max_win, side)
    if win_size % 2 == 0:
        win_size -= 1
    win_size = max(3, win_size)
    return win_size


def calculate_ssim(img1, img2, data_range=255.0):
    """
    SSIM using skimage.metrics.structural_similarity (compare_ssim removed in
    newer scikit-image). Supports HxW grayscale or HxWxC RGB/multichannel.
    """
    img1 = np.asarray(img1, dtype=np.float64)
    img2 = np.asarray(img2, dtype=np.float64)
    assert img1.shape == img2.shape, (
        f'Image shapes differ: {img1.shape}, {img2.shape}.')
    h, w = img1.shape[:2]
    win_size = _adaptive_ssim_win_size(h, w)
    kw = dict(data_range=data_range, win_size=win_size)

    try:
        from skimage.metrics import structural_similarity as ssim_fn
    except ImportError:
        from skimage.measure import compare_ssim as ssim_fn
        if img1.ndim == 3:
            return float(
                ssim_fn(img1, img2, multichannel=True, **kw))
        return float(ssim_fn(img1, img2, **kw))

    if img1.ndim == 3:
        try:
            return float(ssim_fn(img1, img2, channel_axis=-1, **kw))
        except TypeError:
            return float(ssim_fn(img1, img2, multichannel=True, **kw))
    return float(ssim_fn(img1, img2, **kw))


def calc_psnr_and_ssim(img1, img2):
    """Backward-compatible: PSNR + SSIM, images in [0, 255]."""
    return calculate_psnr(img1, img2), calculate_ssim(img1, img2)


def compute_all_frame_metrics(img1, img2, lpips_model=None, lpips_device=None):
    """
    Compute PSNR, SSIM, MSE, MAE, and optionally LPIPS for one frame pair.
    Images: numpy, RGB, uint8 [0,255], shape (H,W,3).

    Returns:
        dict with keys: psnr, ssim, mse, mae, lpips (None if no model).
    """
    mse = calculate_mse(img1, img2)
    mae = calculate_mae(img1, img2)
    psnr = calculate_psnr(img1, img2)
    ssim = calculate_ssim(img1, img2)
    lp = None
    if lpips_model is not None and lpips_device is not None:
        lp = calculate_lpips_distance(
            lpips_model, img1, img2, lpips_device)
    return {
        'psnr': psnr,
        'ssim': ssim,
        'mse': mse,
        'mae': mae,
        'lpips': lp,
    }


# ---------------------------------------------------------------------------
# LPIPS
# ---------------------------------------------------------------------------


def init_lpips_model(net='alex', device=None):
    """
    Load LPIPS model (requires `pip install lpips`).
    net: 'alex' | 'vgg' | 'squeeze'
    """
    try:
        import lpips as lpips_pkg
    except ImportError as e:
        raise ImportError(
            'LPIPS requires the lpips package. Install with: pip install lpips'
        ) from e
    if device is None:
        device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
    model = lpips_pkg.LPIPS(net=net, verbose=False)
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def calculate_lpips_distance(lpips_model, img1, img2, device):
    """
    img1, img2: uint8 numpy (H, W, 3) RGB.
    Returns scalar LPIPS (lower is more similar).
    """
    def _to_tensor(x):
        t = torch.from_numpy(np.asarray(x, dtype=np.float32))
        if t.dim() == 2:
            t = t.unsqueeze(-1).repeat(1, 1, 3)
        t = t.permute(2, 0, 1).unsqueeze(0) / 255.0 * 2.0 - 1.0
        return t.to(device)

    with torch.no_grad():
        d = lpips_model(_to_tensor(img1), _to_tensor(img2))
    return float(d.mean().item())


# ---------------------------------------------------------------------------
# I3D activations & VFID (Frechet distance in feature space)
# ---------------------------------------------------------------------------


def init_i3d_model(i3d_model_path, device=None):
    if device is None:
        device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[Loading I3D model from {i3d_model_path} for VFID ...]')
    i3d_model = InceptionI3d(400, in_channels=3, final_endpoint='Logits')
    state = torch.load(i3d_model_path, map_location=device)
    i3d_model.load_state_dict(state)
    i3d_model.to(device)
    i3d_model.eval()
    return i3d_model


def calculate_i3d_activations(video1, video2, i3d_model, device):
    """
    VFID / I3D features for two videos.
    video1, video2: list[PIL.Image] (same length), RGB.
    Returns:
        (feat1, feat2) 1D numpy vectors (flattened activations per video).
    """
    video1 = to_tensors()(video1).unsqueeze(0).to(device)
    video2 = to_tensors()(video2).unsqueeze(0).to(device)
    video1_activations = get_i3d_activations(
        video1, i3d_model).cpu().numpy().flatten()
    video2_activations = get_i3d_activations(
        video2, i3d_model).cpu().numpy().flatten()
    return video1_activations, video2_activations


def calculate_vfid(real_activations, fake_activations, eps=1e-6):
    """
    Frechet distance between two sets of I3D features (VFID).

    Args:
        real_activations: list of 1D np.ndarray, one per video (length D).
        fake_activations: same.

    Previous bug: np.mean(list_of_arrays, axis=0) does not stack videos;
    must np.stack(..., axis=0) first to get shape (N, D).
    """
    if len(real_activations) == 0 or len(fake_activations) == 0:
        raise ValueError('Empty activations list for VFID.')
    real = np.stack([np.asarray(a, dtype=np.float64).ravel()
                     for a in real_activations], axis=0)
    fake = np.stack([np.asarray(a, dtype=np.float64).ravel()
                     for a in fake_activations], axis=0)
    n_r, d = real.shape
    n_f, d2 = fake.shape
    if d != d2:
        raise ValueError(f'Feature dim mismatch: {d} vs {d2}')
    m1 = np.mean(real, axis=0)
    m2 = np.mean(fake, axis=0)

    if n_r == 1:
        s1 = np.eye(d, dtype=np.float64) * eps
    else:
        s1 = np.cov(real, rowvar=False)
        if s1.ndim == 0:
            s1 = np.eye(d, dtype=np.float64) * eps
        elif s1.shape != (d, d):
            s1 = np.atleast_2d(s1)
            if s1.shape[0] != d:
                s1 = np.eye(d, dtype=np.float64) * eps

    if n_f == 1:
        s2 = np.eye(d, dtype=np.float64) * eps
    else:
        s2 = np.cov(fake, rowvar=False)
        if s2.ndim == 0:
            s2 = np.eye(d, dtype=np.float64) * eps
        elif s2.shape != (d, d):
            s2 = np.atleast_2d(s2)
            if s2.shape[0] != d:
                s2 = np.eye(d, dtype=np.float64) * eps

    # Stabilize diagonal for singular covariances
    s1 = s1 + np.eye(d, dtype=np.float64) * eps
    s2 = s2 + np.eye(d, dtype=np.float64) * eps

    return float(calculate_frechet_distance(m1, s1, m2, s2, eps=eps))


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy Frechet distance between Gaussians (mu, sigma)."""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, (
        'Mean vectors have different lengths')
    assert sigma1.shape == sigma2.shape, (
        'Covariances have different dimensions')

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        print(
            'FID/VFID: singular covariance product; adding eps to diagonal.')
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f'Imaginary component {m}')
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return float(
        diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2)
        - 2.0 * tr_covmean)


def get_i3d_activations(batched_video,
                        i3d_model,
                        target_endpoint='Logits',
                        flatten=True,
                        grad_enabled=False):
    with torch.set_grad_enabled(grad_enabled):
        feat = i3d_model.extract_features(
            batched_video.transpose(1, 2), target_endpoint)
    if flatten:
        feat = feat.view(feat.size(0), -1)
    return feat


# ---------------------------------------------------------------------------
# Legacy: flow EPE (unchanged)
# ---------------------------------------------------------------------------


def calculate_epe(flow1, flow2):
    epe = torch.sum((flow1 - flow2) ** 2, dim=1).sqrt()
    epe = epe.view(-1)
    return epe.mean().item()


# This code is from https://github.com/piergiaj/pytorch-i3d/blob/master/pytorch_i3d.py


class MaxPool3dSamePadding(nn.MaxPool3d):
    def compute_pad(self, dim, s):
        if s % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        return max(self.kernel_size[dim] - (s % self.stride[dim]), 0)

    def forward(self, x):
        (batch, channel, t, h, w) = x.size()
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        x = F.pad(x, pad)
        return super(MaxPool3dSamePadding, self).forward(x)


class Unit3D(nn.Module):
    def __init__(self,
                 in_channels,
                 output_channels,
                 kernel_shape=(1, 1, 1),
                 stride=(1, 1, 1),
                 padding=0,
                 activation_fn=F.relu,
                 use_batch_norm=True,
                 use_bias=False,
                 name='unit_3d'):
        super(Unit3D, self).__init__()

        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.name = name
        self.padding = padding

        self.conv3d = nn.Conv3d(
            in_channels=in_channels,
            out_channels=self._output_channels,
            kernel_size=self._kernel_shape,
            stride=self._stride,
            padding=0,
            bias=self._use_bias)

        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(
                self._output_channels, eps=0.001, momentum=0.01)

    def compute_pad(self, dim, s):
        if s % self._stride[dim] == 0:
            return max(self._kernel_shape[dim] - self._stride[dim], 0)
        return max(self._kernel_shape[dim] - (s % self._stride[dim]), 0)

    def forward(self, x):
        (batch, channel, t, h, w) = x.size()
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        x = F.pad(x, pad)

        x = self.conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x


class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, name):
        super(InceptionModule, self).__init__()

        self.b0 = Unit3D(
            in_channels=in_channels, output_channels=out_channels[0],
            kernel_shape=[1, 1, 1], padding=0,
            name=name + '/Branch_0/Conv3d_0a_1x1')
        self.b1a = Unit3D(
            in_channels=in_channels, output_channels=out_channels[1],
            kernel_shape=[1, 1, 1], padding=0,
            name=name + '/Branch_1/Conv3d_0a_1x1')
        self.b1b = Unit3D(
            in_channels=out_channels[1], output_channels=out_channels[2],
            kernel_shape=[3, 3, 3], name=name + '/Branch_1/Conv3d_0b_3x3')
        self.b2a = Unit3D(
            in_channels=in_channels, output_channels=out_channels[3],
            kernel_shape=[1, 1, 1], padding=0,
            name=name + '/Branch_2/Conv3d_0a_1x1')
        self.b2b = Unit3D(
            in_channels=out_channels[3], output_channels=out_channels[4],
            kernel_shape=[3, 3, 3], name=name + '/Branch_2/Conv3d_0b_3x3')
        self.b3a = MaxPool3dSamePadding(
            kernel_size=[3, 3, 3], stride=(1, 1, 1), padding=0)
        self.b3b = Unit3D(
            in_channels=in_channels, output_channels=out_channels[5],
            kernel_shape=[1, 1, 1], padding=0,
            name=name + '/Branch_3/Conv3d_0b_1x1')
        self.name = name

    def forward(self, x):
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return torch.cat([b0, b1, b2, b3], dim=1)


class InceptionI3d(nn.Module):
    """Inception-v1 I3D architecture."""

    VALID_ENDPOINTS = (
        'Conv3d_1a_7x7',
        'MaxPool3d_2a_3x3',
        'Conv3d_2b_1x1',
        'Conv3d_2c_3x3',
        'MaxPool3d_3a_3x3',
        'Mixed_3b',
        'Mixed_3c',
        'MaxPool3d_4a_3x3',
        'Mixed_4b',
        'Mixed_4c',
        'Mixed_4d',
        'Mixed_4e',
        'Mixed_4f',
        'MaxPool3d_5a_2x2',
        'Mixed_5b',
        'Mixed_5c',
        'Logits',
        'Predictions',
    )

    def __init__(self,
                 num_classes=400,
                 spatial_squeeze=True,
                 final_endpoint='Logits',
                 name='inception_i3d',
                 in_channels=3,
                 dropout_keep_prob=0.5):
        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)

        super(InceptionI3d, self).__init__()
        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze
        self._final_endpoint = final_endpoint
        self.logits = None

        self.end_points = {}
        end_point = 'Conv3d_1a_7x7'
        self.end_points[end_point] = Unit3D(
            in_channels=in_channels, output_channels=64,
            kernel_shape=[7, 7, 7], stride=(2, 2, 2), padding=(3, 3, 3),
            name=name + end_point)
        if self._final_endpoint == end_point:
            self.build()
            return

        end_point = 'MaxPool3d_2a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(
            kernel_size=[1, 3, 3], stride=(1, 2, 2), padding=0)
        if self._final_endpoint == end_point:
            self.build()
            return

        end_point = 'Conv3d_2b_1x1'
        self.end_points[end_point] = Unit3D(
            in_channels=64, output_channels=64, kernel_shape=[1, 1, 1],
            padding=0, name=name + end_point)
        if self._final_endpoint == end_point:
            self.build()
            return

        end_point = 'Conv3d_2c_3x3'
        self.end_points[end_point] = Unit3D(
            in_channels=64, output_channels=192, kernel_shape=[3, 3, 3],
            padding=1, name=name + end_point)
        if self._final_endpoint == end_point:
            self.build()
            return

        end_point = 'MaxPool3d_3a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(
            kernel_size=[1, 3, 3], stride=(1, 2, 2), padding=0)
        if self._final_endpoint == end_point:
            self.build()
            return

        end_point = 'Mixed_3b'
        self.end_points[end_point] = InceptionModule(
            192, [64, 96, 128, 16, 32, 32], name + end_point)
        if self._final_endpoint == end_point:
            self.build()
            return

        end_point = 'Mixed_3c'
        self.end_points[end_point] = InceptionModule(
            256, [128, 128, 192, 32, 96, 64], name + end_point)
        if self._final_endpoint == end_point:
            self.build()
            return

        end_point = 'MaxPool3d_4a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(
            kernel_size=[3, 3, 3], stride=(2, 2, 2), padding=0)
        if self._final_endpoint == end_point:
            self.build()
            return

        end_point = 'Mixed_4b'
        self.end_points[end_point] = InceptionModule(
            128 + 192 + 96 + 64, [192, 96, 208, 16, 48, 64], name + end_point)
        if self._final_endpoint == end_point:
            self.build()
            return

        end_point = 'Mixed_4c'
        self.end_points[end_point] = InceptionModule(
            192 + 208 + 48 + 64, [160, 112, 224, 24, 64, 64], name + end_point)
        if self._final_endpoint == end_point:
            self.build()
            return

        end_point = 'Mixed_4d'
        self.end_points[end_point] = InceptionModule(
            160 + 224 + 64 + 64, [128, 128, 256, 24, 64, 64], name + end_point)
        if self._final_endpoint == end_point:
            self.build()
            return

        end_point = 'Mixed_4e'
        self.end_points[end_point] = InceptionModule(
            128 + 256 + 64 + 64, [112, 144, 288, 32, 64, 64], name + end_point)
        if self._final_endpoint == end_point:
            self.build()
            return

        end_point = 'Mixed_4f'
        self.end_points[end_point] = InceptionModule(
            112 + 288 + 64 + 64, [256, 160, 320, 32, 128, 128],
            name + end_point)
        if self._final_endpoint == end_point:
            self.build()
            return

        end_point = 'MaxPool3d_5a_2x2'
        self.end_points[end_point] = MaxPool3dSamePadding(
            kernel_size=[2, 2, 2], stride=(2, 2, 2), padding=0)
        if self._final_endpoint == end_point:
            self.build()
            return

        end_point = 'Mixed_5b'
        self.end_points[end_point] = InceptionModule(
            256 + 320 + 128 + 128, [256, 160, 320, 32, 128, 128],
            name + end_point)
        if self._final_endpoint == end_point:
            self.build()
            return

        end_point = 'Mixed_5c'
        self.end_points[end_point] = InceptionModule(
            256 + 320 + 128 + 128, [384, 192, 384, 48, 128, 128],
            name + end_point)
        if self._final_endpoint == end_point:
            self.build()
            return

        end_point = 'Logits'
        self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7], stride=(1, 1, 1))
        self.dropout = nn.Dropout(dropout_keep_prob)
        self.logits = Unit3D(
            in_channels=384 + 384 + 128 + 128,
            output_channels=self._num_classes, kernel_shape=[1, 1, 1],
            padding=0, activation_fn=None, use_batch_norm=False,
            use_bias=True, name='logits')

        self.build()

    def replace_logits(self, num_classes):
        self._num_classes = num_classes
        self.logits = Unit3D(
            in_channels=384 + 384 + 128 + 128,
            output_channels=self._num_classes, kernel_shape=[1, 1, 1],
            padding=0, activation_fn=None, use_batch_norm=False,
            use_bias=True, name='logits')

    def build(self):
        for k in self.end_points.keys():
            self.add_module(k, self.end_points[k])

    def forward(self, x):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)

        x = self.logits(self.dropout(self.avg_pool(x)))
        if self._spatial_squeeze:
            logits = x.squeeze(3).squeeze(3)
        return logits

    def extract_features(self, x, target_endpoint='Logits'):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)
                if end_point == target_endpoint:
                    break
        if target_endpoint == 'Logits':
            return x.mean(4).mean(3).mean(2)
        return x

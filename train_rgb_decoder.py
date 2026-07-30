#!/usr/bin/env python
import os
import sys
import argparse
import datetime
from math import exp

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision


def add_paths_for_imports():
    # Prefer the nested GaussianLSS/scripts over the top-level scripts
    here = os.path.dirname(os.path.abspath(__file__))
    pkg_dir = os.path.abspath(os.path.join(here, '..'))          # GaussianLSS/
    repo_root = os.path.abspath(os.path.join(pkg_dir, '..'))     # repository root
    if pkg_dir not in sys.path:
        sys.path.insert(0, pkg_dir)
    if repo_root not in sys.path:
        sys.path.append(repo_root)


add_paths_for_imports()
try:
    from scripts.common import prepare_val  # prefer nested
except Exception:
    # fallback if environment resolves differently
    from GaussianLSS.scripts.common import prepare_val


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.SiLU(inplace=True),
        )

    def forward(self, z):
        return self.net(z)


class UNetLite(nn.Module):
    def __init__(self, in_ch=128, base=64, out_ch=3):
        super().__init__()
        b = base
        # Encoders
        self.enc1 = DoubleConv(in_ch, b)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(b, b * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(b * 2, b * 4)
        self.pool3 = nn.MaxPool2d(2)
        # Bottleneck
        self.bott = DoubleConv(b * 4, b * 4)
        # Decoders with corrected concat channel sizes
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec3 = DoubleConv(b * 4 + b * 4, b * 2)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = DoubleConv(b * 2 + b * 2, b)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = nn.Sequential(
            nn.Conv2d(b * 2, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.SiLU(inplace=True),
            nn.Conv2d(b, out_ch, 1),
        )

    def forward(self, z):
        e1 = self.enc1(z)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bott(self.pool3(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.sigmoid(d1)


class PixelMLP(nn.Module):
    """Per-pixel MLP implemented with 1x1 convs.

    Input:  [B, C_in, H, W]
    Output: [B, C_out, H, W]
    """
    def __init__(self, in_ch: int, hidden: int = 128, out_ch: int = 3, depth: int = 3):
        super().__init__()
        layers = []
        c = in_ch
        for i in range(max(0, depth)):
            h = hidden if i < depth - 1 else hidden
            layers += [nn.Conv2d(c, h, 1, bias=False), nn.BatchNorm2d(h), nn.SiLU(inplace=True)]
            c = h
        layers += [nn.Conv2d(c, out_ch, 1, bias=True)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        y = self.net(x)
        return torch.sigmoid(y)


class RGBMLPDecoder(nn.Module):
    """FeatSplat-style decoder: concatenates a learned per-camera embedding to
    each pixel feature and applies a pixel MLP to output RGB.
    """
    def __init__(self, feat_ch: int, num_cams: int, cam_embed_dim: int = 16, hidden: int = 128, depth: int = 3):
        super().__init__()
        self.cam_emb = nn.Embedding(num_cams, cam_embed_dim)
        self.mlp = PixelMLP(in_ch=feat_ch + cam_embed_dim, hidden=hidden, out_ch=3, depth=depth)

    def forward(self, x_bn: torch.Tensor, cam_idx: torch.Tensor):
        # x_bn: [B*N, D, H, W]; cam_idx: [B*N]
        Bn, D, H, W = x_bn.shape
        e = self.cam_emb(cam_idx).view(Bn, -1, 1, 1).expand(Bn, -1, H, W)
        z = torch.cat([x_bn, e], dim=1)
        return self.mlp(z)


class HybridRGBDecoder(nn.Module):
        """Hybrid CNN+MLP with camera-conditioned gate.

        Inputs:
            - x_in:   [B*N, C_in, H, W]  (may include extras like occ/xy)
            - cam_idx:[B*N]
        Output:
            - rgb:    [B*N, 3, H, W]
        """
        def __init__(self, in_ch: int, num_cams: int, cam_embed_dim: int = 16, mid: int = 96):
                super().__init__()
                g = 8  # group norm groups
                self.conv = nn.Sequential(
                        nn.Conv2d(in_ch, mid, 3, padding=1, bias=False), nn.GroupNorm(max(1, mid // g), mid), nn.SiLU(inplace=True),
                        nn.Conv2d(mid, mid, 3, padding=1, bias=False), nn.GroupNorm(max(1, mid // g), mid), nn.SiLU(inplace=True),
                )
                self.cam_emb = nn.Embedding(num_cams, cam_embed_dim)
                self.gate = nn.Linear(cam_embed_dim, mid)
                self.head = nn.Sequential(
                        nn.Conv2d(mid, mid, 1, bias=False), nn.GroupNorm(max(1, mid // g), mid), nn.SiLU(inplace=True),
                        nn.Conv2d(mid, 3, 1)
                )

        def forward(self, x, cam_idx):
                Bn, _, H, W = x.shape
                z = self.conv(x)
                e = self.cam_emb(cam_idx)                  # [Bn, E]
                g = torch.sigmoid(self.gate(e)).view(Bn, -1, 1, 1).expand_as(z)
                z = z * g
                y = self.head(z)
                return torch.sigmoid(y)


def gaussian(window_size, sigma):
    gauss = torch.tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)],
                         dtype=torch.float32)
    return gauss / gauss.sum()


def create_window(window_size, channel, device):
    _1D = gaussian(window_size, 1.5).to(device)
    _2D = (_1D[:, None] @ _1D[None, :]).unsqueeze(0).unsqueeze(0)
    window = _2D.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(x, y, window_size=11, val_range=1.0):
    # Compute SSIM in float32 for numerical stability and AMP compatibility
    x = x.float()
    y = y.float()
    padd = window_size // 2
    channel = x.size(1)
    window = create_window(window_size, channel, x.device)
    mu1 = nn.functional.conv2d(x, window, padding=padd, groups=channel)
    mu2 = nn.functional.conv2d(y, window, padding=padd, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = nn.functional.conv2d(x * x, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = nn.functional.conv2d(y * y, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = nn.functional.conv2d(x * y, window, padding=padd, groups=channel) - mu1_mu2
    C1 = (0.01 * val_range) ** 2
    C2 = (0.03 * val_range) ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def psnr(a, b, eps=1e-8):
    mse = torch.mean((a - b) ** 2)
    return 20.0 * torch.log10(1.0 / torch.sqrt(mse + eps))


def to_device_batch(batch, device):
    for k, v in list(batch.items()):
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device, non_blocking=True)
        elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
            batch[k] = [t.to(device, non_blocking=True) for t in v]
    return batch


def l2_normalize_feats(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # x: [B,C,H,W]
    denom = torch.clamp(x.pow(2).sum(1, keepdim=True).sqrt(), min=eps)
    return x / denom


def make_xy_channels(H: int, W: int, device: torch.device) -> torch.Tensor:
    ys = torch.linspace(-1.0, 1.0, steps=H, device=device)
    xs = torch.linspace(-1.0, 1.0, steps=W, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    return torch.stack([xx, yy], dim=0)  # [2,H,W]


def densify_neighbor_mean(x: torch.Tensor, valid: torch.Tensor, iters: int = 2) -> torch.Tensor:
    """Fill invalid pixels by neighbor mean; expands validity over iterations.
    x: [B,C,H,W]; valid: [B,1,H,W] in {0,1}
    """
    if iters <= 0:
        return x
    B, C, H, W = x.shape
    weight = torch.ones(1, 1, 3, 3, device=x.device)
    for _ in range(iters):
        num = F.conv2d(x * valid, weight.expand(C, 1, 3, 3), padding=1, groups=C)
        den = F.conv2d(valid, weight, padding=1)
        avg = num / (den.clamp_min(1e-6))
        x = torch.where(valid.bool().expand_as(x), x, avg)
        # expand validity where any neighbor was valid
        valid = (F.conv2d(valid, weight, padding=1) > 0).float()
    return x


def tv_loss(img: torch.Tensor) -> torch.Tensor:
    """Total variation loss on [B,3,H,W] or [B,C,H,W]."""
    dh = (img[..., 1:, :] - img[..., :-1, :]).abs().mean()
    dw = (img[..., :, 1:] - img[..., :, :-1]).abs().mean()
    return dh + dw


class VGGPerceptual(nn.Module):
    def __init__(self, layers=(2, 7, 12, 21), requires_grad=False):
        super().__init__()
        vgg = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1).features
        self.slices = nn.ModuleList()
        prev = 0
        for l in layers:
            self.slices.append(nn.Sequential(*[vgg[i] for i in range(prev, l)]))
            prev = l
        for s in self.slices:
            for p in s.parameters():
                p.requires_grad = requires_grad
        self.register_buffer('mean', torch.tensor([0.485,0.456,0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.tensor([0.229,0.224,0.225]).view(1,3,1,1))

    def forward(self, x, y):
        # inputs in [0,1], compute in float32 for stability
        x = x.float()
        y = y.float()
        x_n = (x - self.mean) / self.std
        y_n = (y - self.mean) / self.std
        loss = 0.0
        for s in self.slices:
            x_n = s(x_n)
            y_n = s(y_n)
            loss = loss + F.l1_loss(x_n, y_n)
        return loss


def main():
    parser = argparse.ArgumentParser(description='Train RGB decoder from perspective features')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to GaussianLSS checkpoint .ckpt')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--save-interval', type=int, default=5)
    parser.add_argument('--save-dir', type=str, default=None)
    parser.add_argument('--max-steps-per-epoch', type=int, default=0, help='Limit steps per epoch for debug (0 = no cap)')
    parser.add_argument('--decoder-type', type=str, default='hybrid', choices=['mlp', 'unet', 'hybrid'], help='Decoder architecture')
    parser.add_argument('--cam-embed-dim', type=int, default=16, help='Camera embedding dimension for MLP decoder')
    parser.add_argument('--use-densify', action='store_true', help='Densify sparse perspective features')
    parser.add_argument('--densify-iters', type=int, default=2, help='Neighbor-mean iterations for densification')
    parser.add_argument('--add-xy', action='store_true', help='Append XY positional channels')
    parser.add_argument('--add-occ', action='store_true', help='Append occupancy channel')
    parser.add_argument('--l2-norm-feats', action='store_true', help='L2-normalize feature vectors per pixel')
    parser.add_argument('--tv-weight', type=float, default=0.01, help='Weight for TV loss')
    parser.add_argument('--w-ssim', type=float, default=0.5, help='Weight for SSIM term')
    parser.add_argument('--use-alpha-mask', action='store_true', help='Weight losses by renderer alpha coverage (valid px)')
    parser.add_argument('--alpha-eps', type=float, default=1e-4, help='Alpha threshold for valid mask')
    parser.add_argument('--use-perc', action='store_true', help='Add VGG19 perceptual loss')
    parser.add_argument('--perc-weight', type=float, default=0.1, help='Weight for perceptual loss')
    # Performance / focus controls
    parser.add_argument('--amp', action='store_true', help='Use mixed precision for decoder training')
    parser.add_argument('--cam-sample-k', type=int, default=0, help='Subsample K cameras per sample after render (0: use all)')
    parser.add_argument('--compress-dim', type=int, default=0, help='1x1 conv feature compression before decoder (0: off)')
    parser.add_argument('--persp-scale', type=float, default=1.0, help='Downscale perspective render resolution in backbone model (<=1.0)')
    parser.add_argument('--persp-cam-k', type=int, default=0, help='Ask backbone model to render only K cameras per step (0: all)')
    # Strictly use perspective_sub and train on all cameras (no cam-index)
    parser.add_argument('--use-raw-calib', action='store_true', help='Use raw intrinsics/extrinsics if available')
    args = parser.parse_args()

    device = torch.device(args.device)

    # Prepare model and loader
    overrides = [
        'model.enable_perspective=True',
        'data.split_intrin_extrin=True'
    ]
    # Lightweight perspective settings to speed up gsplat rendering
    if args.persp_scale and args.persp_scale != 1.0:
        overrides.append(f'model.persp_scale={float(args.persp_scale)}')
    if args.persp_cam_k and int(args.persp_cam_k) > 0:
        overrides.append(f'model.persp_cam_sample_k={int(args.persp_cam_k)}')
    model, network, loader, viz, dataset = prepare_val('GaussianLSS', device, args.checkpoint,
                                                       overrides=overrides, mode='train', batch_size=args.batch_size)

    # Freeze base model
    for p in network.parameters():
        p.requires_grad = False
    network.eval()

    # Output dir
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    save_root = args.save_dir or os.path.abspath(os.path.join(os.path.dirname(__file__), 'outputs', f'rgb_decoder_{ts}'))
    os.makedirs(save_root, exist_ok=True)
    print('Saving to:', save_root)

    rgb_dec = None
    num_cams_for_emb = None
    opt = None
    perc = VGGPerceptual().to(device) if args.use_perc else None
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and (device.type == 'cuda'))
    preproj = None

    w_l1, w_ssim = 1.0, float(args.w_ssim)

    for epoch in range(1, args.epochs + 1):
        total_l1, total_psnr, total_ssim, steps = 0.0, 0.0, 0.0, 0
        for batch in loader:
            batch = to_device_batch(batch, device)

            # Optionally use raw calib for denser perspective
            if args.use_raw_calib and 'intrinsics_raw' in batch and 'extrinsics_raw' in batch:
                batch['intrinsics'] = batch['intrinsics_raw']
                batch['extrinsics'] = batch['extrinsics_raw']

            # Ensure perspective path is rendered by the model
            batch['render_persp'] = True

            # Reduce sparsity
            try:
                if hasattr(network, 'persp_render'):
                    network.persp_render.threshold = 0.0
                if hasattr(network, 'gs_render'):
                    network.gs_render.threshold = 0.002
            except Exception:
                pass

            # Forward base model to get perspective features
            with torch.no_grad():
                pred = network(batch)

            if not isinstance(pred, dict):
                continue

            # Strictly prefer perspective_sub; allow common aliases as fallback
            cand_sub = [
                'perspective_sub', 'persp_sub', 'perspective_sub_full', 'perspective_sub_feat'
            ]
            x_bn = None
            for k in cand_sub:
                if k in pred and pred[k] is not None:
                    x_bn = pred[k]
                    if k != 'perspective_sub':
                        print(f"[info] Using sub features from key='{k}'")
                    break
            if x_bn is None:
                # final fallback to legacy 'perspective' with a warning
                if 'perspective' in pred and pred['perspective'] is not None:
                    x_bn = pred['perspective']
                    print("[warn] 'perspective_sub' not found; falling back to 'perspective'. Consider updating model outputs.")
                else:
                    print('Warning: no perspective_sub-like key in pred; skipping batch')
                    continue

            # Train on all cameras (optionally subsample K cams):
            # reshape [B, N, D, H, W] -> [B*N, D, H, W]
            y_bn = batch['image'].float()                   # [B, N, 3, H, W]
            if x_bn.ndim != 5 or y_bn.ndim != 5:
                print('Unexpected tensor dims; skipping batch')
                continue
            B, N, D, H, W = x_bn.shape
            # Optional per-sample camera subsampling based on alpha coverage
            if args.cam_sample_k and args.cam_sample_k > 0 and args.cam_sample_k < N:
                keep_lists = []
                if args.use_alpha_mask and ('perspective_meta' in pred):
                    metas = pred['perspective_meta']
                    for bi in range(B):
                        cams = metas[bi]
                        scores = []
                        for ni in range(min(N, len(cams))):
                            a = cams[ni].get('alpha', None)
                            if isinstance(a, torch.Tensor):
                                scores.append(float(a.mean()))
                            else:
                                scores.append(0.0)
                        idx = torch.tensor(scores).topk(args.cam_sample_k).indices.tolist()
                        keep_lists.append(sorted(idx))
                else:
                    base = list(range(N))
                    keep = base[: args.cam_sample_k]
                    keep_lists = [keep for _ in range(B)]
                xs, ys = [], []
                for bi in range(B):
                    idxs = keep_lists[bi]
                    xs.append(x_bn[bi, idxs])  # [k,D,H,W]
                    ys.append(y_bn[bi, idxs])  # [k,3,H,W]
                x_bn = torch.stack(xs, dim=0)  # [B,k,D,H,W]
                y_bn = torch.stack(ys, dim=0)  # [B,k,3,H,W]
                N = x_bn.shape[1]
            x = x_bn.reshape(B * N, D, H, W)
            Hy, Wy = y_bn.shape[-2], y_bn.shape[-1]
            y = y_bn.reshape(B * N, 3, Hy, Wy)
            if (Hy != H) or (Wy != W):
                y = F.interpolate(y, size=(H, W), mode='bilinear', align_corners=False)

            # Optional alpha coverage mask from perspective meta
            alpha_w = None
            if args.use_alpha_mask and ('perspective_meta' in pred):
                try:
                    metas = pred['perspective_meta']  # list len B of list len N
                    alpha_maps = []
                    for bi in range(B):
                        cams = metas[bi]
                        for ni in range(min(N, len(cams))):
                            a = cams[ni].get('alpha', None)
                            if isinstance(a, torch.Tensor):
                                alpha_maps.append(a.to(device))  # [H,W]
                            else:
                                alpha_maps.append(torch.ones((H, W), device=device))
                    if len(alpha_maps) == B * N:
                        alpha_w = torch.stack(alpha_maps, dim=0).unsqueeze(1)  # [B*N,1,H,W]
                        # binarize/threshold to avoid tiny weights, then renormalize
                        alpha_w = (alpha_w >= float(args.alpha_eps)).float()
                except Exception:
                    alpha_w = None

            # Build extras and preprocessing
            x_in = x
            # occupancy from non-zero magnitude
            occ = (x_in.abs().sum(1, keepdim=True) > 0).float()
            if args.l2_norm_feats:
                x_in = l2_normalize_feats(x_in)
            if args.use_densify:
                x_in = densify_neighbor_mean(x_in, occ, iters=max(1, int(args.densify_iters)))
            extras = []
            if args.add_occ:
                extras.append(occ)
            if args.add_xy:
                xy = make_xy_channels(H, W, device=x.device).unsqueeze(0).repeat(B * N, 1, 1, 1)
                extras.append(xy)
            if len(extras) > 0:
                x_in = torch.cat([x_in] + extras, dim=1)

            if rgb_dec is None:
                D_in = x_in.shape[1]
                num_cams_for_emb = N
                # Optional channel compression
                if args.compress_dim and args.compress_dim > 0 and args.compress_dim < D_in:
                    preproj = nn.Sequential(
                        nn.Conv2d(D_in, args.compress_dim, 1, bias=False),
                        nn.GroupNorm(max(1, args.compress_dim // 8), args.compress_dim),
                        nn.SiLU(inplace=True),
                    ).to(device)
                    D_in = args.compress_dim
                if args.decoder_type == 'unet':
                    rgb_dec = UNetLite(in_ch=D_in, base=48, out_ch=3).to(device)
                elif args.decoder_type == 'mlp':
                    rgb_dec = RGBMLPDecoder(feat_ch=D_in, num_cams=num_cams_for_emb, cam_embed_dim=args.cam_embed_dim, hidden=96, depth=3).to(device)
                else:  # hybrid
                    rgb_dec = HybridRGBDecoder(in_ch=D_in, num_cams=num_cams_for_emb, cam_embed_dim=args.cam_embed_dim, mid=80).to(device)
                params = list(rgb_dec.parameters()) + (list(preproj.parameters()) if preproj is not None else [])
                opt = optim.AdamW(params, lr=1e-3, weight_decay=1e-4)

            # Train step
            rgb_dec.train()
            if preproj is not None:
                preproj.train()
            opt.zero_grad(set_to_none=True)
            # Build cam indices [0..N-1] for each sample, flattened to [B*N]
            cam_idx = torch.arange(N, device=device, dtype=torch.long).unsqueeze(0).expand(B, N).reshape(-1)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled() if scaler is not None else False):
                xin_use = preproj(x_in) if preproj is not None else x_in
                if args.decoder_type == 'unet':
                    y_hat = rgb_dec(xin_use)
                elif args.decoder_type == 'mlp':
                    y_hat = rgb_dec(xin_use, cam_idx)
                else:
                    y_hat = rgb_dec(xin_use, cam_idx)
            if alpha_w is not None:
                l1_map = (y_hat - y).abs().mean(1, keepdim=True)  # [B*N,1,H,W]
                # Normalize weights to keep loss scale stable
                w = alpha_w
                denom = w.mean().clamp_min(1e-6)
                l1 = (l1_map * w).mean() / denom
                # SSIM on masked region: approximate by multiplying maps before mean
                ssim_map = ssim(y_hat.clamp(0, 1), y.clamp(0, 1))  # returns scalar by default
                # Fallback: compute global SSIM when masked SSIM isn't directly supported
                ssim_val = ssim_map
            else:
                l1 = nn.functional.l1_loss(y_hat, y)
                ssim_val = ssim(y_hat.clamp(0, 1), y.clamp(0, 1))
            loss = w_l1 * l1 + w_ssim * (1.0 - ssim_val)
            if args.tv_weight > 0:
                loss = loss + float(args.tv_weight) * tv_loss(y_hat)
            if perc is not None and args.perc_weight > 0:
                loss = loss + float(args.perc_weight) * perc(y_hat.clamp(0,1), y.clamp(0,1))
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward(); opt.step()

            with torch.no_grad():
                cur_psnr = psnr(y_hat.clamp(0, 1), y)
            total_l1 += float(l1.detach().cpu())
            total_psnr += float(cur_psnr.detach().cpu())
            total_ssim += float(ssim_val.detach().cpu())
            steps += 1

            # Periodic progress
            if (steps % 25) == 1:
                print(f"  epoch {epoch:02d} step {steps}: L1={total_l1/steps:.4f} SSIM={total_ssim/steps:.3f} PSNR={total_psnr/steps:.2f}")

            # Optional cap to avoid long/debug runs stalling
            if args.max_steps_per_epoch and steps >= args.max_steps_per_epoch:
                break

        if steps == 0:
            print(f"Epoch {epoch:02d}: no valid batches (missing perspective).")
            continue

        print(f"Epoch {epoch:02d}/{args.epochs} | L1={total_l1/steps:.4f} | SSIM={total_ssim/steps:.3f} | PSNR={total_psnr/steps:.2f} dB")

        # Save periodically
        if (epoch % args.save_interval == 0) or (epoch == args.epochs):
            ckpt_path = os.path.join(save_root, f'decoder_epoch{epoch:02d}.pth')
            torch.save({'epoch': epoch, 'state_dict': rgb_dec.state_dict()}, ckpt_path)

            # Save small visualization grid from the last seen batch
            try:
                os.makedirs(os.path.join(save_root, f'epoch_{epoch:02d}'), exist_ok=True)
                rgb_dec.eval()
                with torch.no_grad():
                    xin_use = preproj(x_in) if preproj is not None else x_in
                    if args.decoder_type == 'unet':
                        preds = rgb_dec(xin_use).clamp(0, 1).detach().cpu()
                    elif args.decoder_type in ('mlp', 'hybrid'):
                        preds = rgb_dec(xin_use, cam_idx).clamp(0, 1).detach().cpu()
                    else:
                        preds = rgb_dec(xin_use).clamp(0, 1).detach().cpu()
                import matplotlib.pyplot as plt
                import numpy as np
                out_dir = os.path.join(save_root, f'epoch_{epoch:02d}')
                num_show = min(x.shape[0], 6)
                for i in range(num_show):
                    tgt_np = y[i].detach().cpu().permute(1, 2, 0).numpy()
                    rec_np = preds[i].permute(1, 2, 0).numpy()
                    err = (np.abs(rec_np - tgt_np)).mean(axis=2)

                    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
                    axes[0].imshow(tgt_np); axes[0].set_title(f'Target (b{i})'); axes[0].axis('off')
                    axes[1].imshow(rec_np); axes[1].set_title(f'Recon (b{i})'); axes[1].axis('off')
                    im = axes[2].imshow(err, cmap='inferno'); axes[2].set_title('Mean |error|'); axes[2].axis('off')
                    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
                    fig.tight_layout()
                    fig.savefig(os.path.join(out_dir, f'sample_{i}.png'), dpi=120)
                    plt.close(fig)
                # Also save a small tensor snapshot of predictions and targets
                torch.save({
                    'preds': preds[:num_show],
                    'targets': y[:num_show].detach().cpu()
                }, os.path.join(out_dir, 'predictions.pt'))
                print(f"Saved checkpoint and samples at epoch {epoch:02d} -> {ckpt_path}")
            except Exception as e:
                print('Visualization save failed:', e)


if __name__ == '__main__':
    main()

import os
import sys
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure repository root is on sys.path so imports like GaussianLSS.scripts.common work
THIS_FILE = os.path.abspath(__file__)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(THIS_FILE)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Also add inner package folder (project/GaussianLSS) so imports like
# `from GaussianLSS.common import ...` resolve to the correct module when
# the repo contains a nested `GaussianLSS/` package directory.
INNER_PKG = os.path.join(REPO_ROOT, 'GaussianLSS')
if os.path.isdir(INNER_PKG) and INNER_PKG not in sys.path:
    sys.path.insert(0, INNER_PKG)

from GaussianLSS.scripts.common import prepare_val

import argparse
import numpy as np
from tqdm import tqdm

def finite_mean_std(t: torch.Tensor, dims, keepdim=True, eps: float = 1e-6):
    """Compute mean/std over dims while ignoring non-finite entries.

    Falls back to standard mean/std if all values are finite. Compatible with
    PyTorch versions lacking torch.nanmean/nanstd.
    """
    mask = torch.isfinite(t)
    if not mask.any():
        # no finite values; return zeros to avoid NaNs
        mean = torch.zeros(*[1 if keepdim else s for i, s in enumerate(t.shape) if i in dims], device=t.device, dtype=t.dtype)
        std = torch.ones_like(mean) * eps
        return mean, std
    count = mask.sum(dim=dims, keepdim=keepdim).clamp_min(1)
    t0 = torch.where(mask, t, torch.zeros_like(t))
    mean = t0.sum(dim=dims, keepdim=keepdim) / count
    var = torch.where(mask, (t - mean) ** 2, torch.zeros_like(t)).sum(dim=dims, keepdim=keepdim) / count
    std = torch.sqrt(var + eps)
    return mean, std


class ASPP(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, rates=(1, 2, 3, 6), groups: int = 32):
        super().__init__()
        branches = []
        for r in rates:
            branches.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3 if r > 1 else 1, padding=r if r > 1 else 0, dilation=r if r > 1 else 1, bias=False),
                    nn.GroupNorm(min(groups, out_ch), out_ch),
                    nn.SiLU(inplace=True),
                )
            )
        self.branches = nn.ModuleList(branches)
        self.proj = nn.Sequential(
            nn.Conv2d(out_ch * len(rates), out_ch, kernel_size=1, bias=False),
            nn.GroupNorm(min(groups, out_ch), out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        feats = [b(x) for b in self.branches]
        y = torch.cat(feats, dim=1)
        return self.proj(y)


class ASPPDownMapper(nn.Module):
    """ASPP + stride-2 downsamples to reach target scale, then 1x1 to out_ch.

    We build the downsampling depth once using the first seen spatial sizes.
    """
    def __init__(self, in_ch: int, out_ch: int, base: int = 192, groups: int = 32, down_steps: int = 3):
        super().__init__()
        b = max(64, min(base, 512))
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, b, 3, padding=1, bias=False),
            nn.GroupNorm(min(groups, b), b),
            nn.SiLU(inplace=True),
        )
        self.aspp = ASPP(b, b, rates=(1, 2, 3, 6), groups=groups)
        downs = []
        for _ in range(max(0, down_steps)):
            downs += [
                nn.Conv2d(b, b, 3, stride=2, padding=1, bias=False),
                nn.GroupNorm(min(groups, b), b),
                nn.SiLU(inplace=True),
            ]
        self.down = nn.Sequential(*downs) if downs else nn.Identity()
        self.head = nn.Sequential(
            nn.Conv2d(b, b, 3, padding=1, bias=False),
            nn.GroupNorm(min(groups, b), b),
            nn.SiLU(inplace=True),
            nn.Conv2d(b, out_ch, 1, bias=True),
        )

    def forward(self, x):
        y = self.stem(x)
        y = self.aspp(y)
        y = self.down(y)
        y = self.head(y)
        return y


def pick_highest_res(feat_struct):
    if isinstance(feat_struct, torch.Tensor):
        return feat_struct
    if isinstance(feat_struct, (list, tuple)) and len(feat_struct) > 0:
        best = max([t for t in feat_struct if isinstance(t, torch.Tensor)], key=lambda t: t.shape[-2]*t.shape[-1])
        return best
    if isinstance(feat_struct, dict) and len(feat_struct) > 0:
        best_k = max(feat_struct.keys(), key=lambda k: feat_struct[k].shape[-2]*feat_struct[k].shape[-1])
        return feat_struct[best_k]
    return None


def get_neck_feature_for_cam(network, imgs, device):
    # imgs: [B, C, H, W]
    with torch.no_grad():
        xin = imgs
        if hasattr(network, 'normalize') and callable(getattr(network, 'normalize')):
            try:
                xin = network.normalize(xin)
            except Exception:
                xin = xin.float()
        z_back = None
        if hasattr(network, 'backbone') and callable(getattr(network, 'backbone')):
            try:
                z_back = network.backbone(xin)
            except Exception:
                z_back = None
        z_neck = None
        if z_back is not None and hasattr(network, 'neck') and callable(getattr(network, 'neck')):
            try:
                z_neck = network.neck(z_back)
            except Exception:
                z_neck = None
        backs = []
        if isinstance(z_neck, torch.Tensor):
            backs = [z_neck]
        elif isinstance(z_neck, (list, tuple)):
            backs = [t for t in z_neck if isinstance(t, torch.Tensor)]
        elif isinstance(z_neck, dict):
            backs = [v for v in z_neck.values() if isinstance(v, torch.Tensor)]
        if not backs:
            if isinstance(z_back, torch.Tensor):
                backs = [z_back]
            elif isinstance(z_back, (list, tuple)):
                backs = [t for t in z_back if isinstance(t, torch.Tensor)]
            elif isinstance(z_back, dict):
                backs = [v for v in z_back.values() if isinstance(v, torch.Tensor)]
        assert len(backs) > 0, "No encoder/neck tensors available to build target."
        # pick highest-res
        target = pick_highest_res(backs)
        return target


def train(args):
    device = torch.device(args.device if args.device is not None else ('cuda:0' if torch.cuda.is_available() else 'cpu'))
    overrides = ['model.enable_perspective=True', 'data.split_intrin_extrin=True']

    # Prepare dataset and load the full GaussianLSS model with the Stage 1 checkpoint
    model, network, loader, viz, dataset = prepare_val('GaussianLSS', device, args.checkpoint, overrides=overrides, mode='train', batch_size=args.batch_size)
    print(f"Using dataset with {len(dataset)} samples | loader batches (approx) = {len(loader)} | batch_size={args.batch_size}")
    if len(dataset) == 0:
        raise RuntimeError('Dataset appears empty after prepare_val; check dataset paths and split names')

    # Freeze base model and network
    model.eval()
    network.eval()
    for p in model.parameters():
        p.requires_grad = False
    for p in network.parameters():
        p.requires_grad = False

    # Decoder will be created once we infer input/output channels from a batch
    decoder = None
    opt = None
    cos_sim = nn.CosineSimilarity(dim=1)

    it = 0
    start = time.time()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        it_epoch = 0
        samples_seen = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}", total=len(loader))
        for batch in pbar:
            # optional limit
            if args.max_batches is not None and it_epoch >= args.max_batches:
                break

            # move batch tensors to device where needed
            for k, v in list(batch.items()):
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
                elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
                    batch[k] = [t.to(device) for t in v]

            # Get perspective_sub from network (eval, no grad) - decoder input must be perspective_sub
            with torch.no_grad():
                pred = network(batch)

            # Prefer explicit perspective_sub; fall back to legacy key 'perspective'
            psub = pred.get('perspective_sub', pred.get('perspective', None))
            if psub is None:
                raise RuntimeError('perspective_sub not found in network outputs; ensure model.enable_perspective=True via overrides')
            # shape: [B, N, D, H, W]
            B, N, D, H_in, W_in = psub.shape

            # inputs stacked cam-major (we feed the decoder camera-wise inputs)
            inputs = [psub[:, cam] for cam in range(N)]
            inp_stack = torch.cat(inputs, dim=0)  # [B*N, D, H_in, W_in]
            # Build targets directly from network._debug_backbone_features (features[0])
            target_feat = getattr(network, '_debug_backbone_features', None)
            if target_feat is None:
                raise RuntimeError('Network did not expose _debug_backbone_features; please ensure forward ran and backbone produced features.')

            # target_feat expected shape: [B, N, C_tgt, H_tgt, W_tgt]
            if target_feat.dim() == 5:
                Bt2, Nt2, Ct2, Ht2, Wt2 = target_feat.shape
                tgt_stack = target_feat.reshape(Bt2 * Nt2, Ct2, Ht2, Wt2)
            elif target_feat.dim() == 4:
                # Already flattened per-camera
                tgt_stack = target_feat
            else:
                raise RuntimeError(f'Unexpected target feature shape: {tuple(target_feat.shape)}')

            C_tgt = tgt_stack.shape[1]

            # Create decoder if not yet created; input channels = D, output channels = C_tgt
            if decoder is None:
                mapper_base = getattr(args, 'mapper_base', 256)
                # decide downsample steps based on H_in -> H_tgt ratio (round to nearest power-of-two steps)
                H_tgt, W_tgt = tgt_stack.shape[-2], tgt_stack.shape[-1]
                # Avoid log of zero
                eps = 1e-6
                ds_h = max(0, int(round(math.log2(max(H_in, eps) / max(H_tgt, eps)))))
                ds_w = max(0, int(round(math.log2(max(W_in, eps) / max(W_tgt, eps)))))
                down_steps = max(ds_h, ds_w)
                decoder = ASPPDownMapper(in_ch=D, out_ch=C_tgt, base=mapper_base, groups=args.groups, down_steps=down_steps).to(device)
                opt = torch.optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=1e-4)

            # Sanity: decoder output channels should match target channels
            try:
                outC = decoder.net[-1].out_channels
                if C_tgt != outC:
                    raise RuntimeError(f'Decoder output channels {outC} != target channels {C_tgt}')
            except Exception:
                pass

            # Forward decoder
            decoder.train()
            y_hat = decoder(inp_stack)

            # Ensure spatial size matches target; upsample decoder output if needed
            tgt_h, tgt_w = tgt_stack.shape[-2:]
            if y_hat.shape[-2:] != (tgt_h, tgt_w):
                y_hat = F.interpolate(y_hat, size=(tgt_h, tgt_w), mode='bilinear', align_corners=False)
                if it == 0:
                    print(f'[diagnostic] interpolated decoder output from {inp_stack.shape[-2:]} -> {(tgt_h, tgt_w)}')

            # Optional feature normalization to mitigate scale mismatch
            if getattr(args, 'normalize_features', True):
                def norm_feat(ti):
                    mean, std = finite_mean_std(ti, dims=[1,2,3], keepdim=True)
                    return (ti - mean) / (std + 1e-6)
                y_cmp = norm_feat(y_hat)
                t_cmp = norm_feat(tgt_stack)
            else:
                y_cmp, t_cmp = y_hat, tgt_stack

            # Combined loss: L1 + SmoothL1 + cosine distance
            mask = torch.isfinite(t_cmp)
            if mask.any():
                l1 = F.l1_loss(y_cmp[mask.expand_as(y_cmp)], t_cmp[mask.expand_as(t_cmp)])
                hub = F.smooth_l1_loss(y_cmp[mask.expand_as(y_cmp)], t_cmp[mask.expand_as(t_cmp)], beta=0.5)
            else:
                l1 = F.l1_loss(y_cmp, t_cmp)
                hub = F.smooth_l1_loss(y_cmp, t_cmp, beta=0.5)
            # cosine over channels at each pixel
            try:
                cos = cos_sim(y_cmp, t_cmp)  # [B, H, W]
                cos_loss = 1.0 - torch.nanmean(cos)
            except Exception:
                cos_loss = torch.tensor(0.0, device=device)

            # foreground-weighted L1 to emphasize object pixels (configurable)
            fg_weight = float(getattr(args, 'fg_weight', 0.7))
            fg_mult = float(getattr(args, 'fg_thresh_mult', 0.5))
            # compute per-sample mean/std to detect foreground
            try:
                mean, std = finite_mean_std(t_cmp, dims=[1,2,3], keepdim=True)
                thr = mean + fg_mult * std
                fg_mask = (t_cmp > thr) & mask
                if fg_mask.any():
                    l1_fg = F.l1_loss(y_cmp[fg_mask.expand_as(y_cmp)], t_cmp[fg_mask.expand_as(t_cmp)])
                    base = (1.0 - fg_weight) * l1 + fg_weight * l1_fg
                else:
                    base = l1
            except Exception:
                base = l1

            lw = float(getattr(args, 'l1_weight', 1.0))
            hw = float(getattr(args, 'huber_weight', 0.5))
            cw = float(getattr(args, 'cosine_weight', 0.1))
            loss = lw * base + hw * hub + cw * cos_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            it += 1
            it_epoch += 1
            samples_seen += inp_stack.shape[0]
            epoch_loss += float(loss.item())

            # update progress bar periodically
            if it % args.log_interval == 0:
                elapsed = time.time() - start
                pbar.set_postfix(loss=f"{loss.item():.4f}", l1=f"{l1.item():.4f}", samples=f"{samples_seen}", elapsed=f"{elapsed:.1f}s")
            if args.max_iters is not None and it >= args.max_iters:
                break

        # end of epoch
        avg = epoch_loss / max(1, it_epoch)
        print(f"Epoch {epoch+1} finished | avg loss {avg:.6f} | samples_seen {samples_seen}")

        # save checkpoint every save_interval epochs or at final epoch
        if ((epoch + 1) % args.save_interval == 0) or (epoch == args.epochs - 1):
            os.makedirs(args.save_dir, exist_ok=True)
            ckpt_path = os.path.join(args.save_dir, f"decoder_epoch{epoch+1}.pth")
            torch.save({'decoder_state': decoder.state_dict(), 'opt_state': opt.state_dict(), 'epoch': epoch+1}, ckpt_path)
            print('Saved decoder checkpoint to', ckpt_path)

            # Save a small tensor dump for inspection (first 2 samples)
            try:
                dump_path = os.path.join(args.save_dir, f"pred_dump_epoch{epoch+1}.npz")
                with torch.no_grad():
                    y_dump = decoder(inp_stack[:2])
                    if y_dump.shape[-2:] != tgt_stack.shape[-2:]:
                        y_dump = F.interpolate(y_dump, size=tgt_stack.shape[-2:], mode='bilinear', align_corners=False)
                np.savez_compressed(
                    dump_path,
                    pred=y_dump.detach().cpu().numpy(),
                    target=tgt_stack[:2].detach().cpu().numpy(),
                )
                print('Saved prediction dump to', dump_path)
            except Exception as e:
                print('[warning] Failed to save prediction dump:', str(e))

        # per-save-interval visuals
        if ((epoch + 1) % args.save_interval == 0) or (epoch == args.epochs - 1):
            try:
                vis_dir = os.path.join(args.save_dir, 'visuals')
                os.makedirs(vis_dir, exist_ok=True)
                decoder.eval()
                with torch.no_grad():
                    y_hat_vis = decoder(inp_stack)
                # Ensure visualization prediction has same spatial size as target (same upsample used in training)
                tgt_h, tgt_w = tgt_stack.shape[-2:]
                if y_hat_vis.shape[-2:] != (tgt_h, tgt_w):
                    orig_size = tuple(y_hat_vis.shape[-2:])
                    y_hat_vis = F.interpolate(y_hat_vis, size=(tgt_h, tgt_w), mode='bilinear', align_corners=False)
                    if it == 0:
                        print(f'[diagnostic] interpolated visualization decoder output from {orig_size} -> {(tgt_h, tgt_w)}')

                inp_cpu = inp_stack.detach().cpu()
                tgt_cpu = tgt_stack.detach().cpu()
                pred_cpu = y_hat_vis.detach().cpu()

                n_vis = min(4, inp_cpu.shape[0])
                import matplotlib.pyplot as plt
                for i in range(n_vis):
                    t = tgt_cpu[i]
                    p = pred_cpu[i]
                    # finite-safe mean over channels
                    t_mean = torch.where(torch.isfinite(t), t, torch.zeros_like(t)).mean(dim=0)
                    p_mean = torch.where(torch.isfinite(p), p, torch.zeros_like(p)).mean(dim=0)
                    diff = (t_mean - p_mean).abs()

                    def norm01(x):
                        x = x.numpy().astype(np.float32)
                        m = np.nanmin(x)
                        M = np.nanmax(x)
                        return (x - m) / (M - m + 1e-6)

                    t_v = norm01(t_mean)
                    p_v = norm01(p_mean)
                    d_v = norm01(diff)

                    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                    axes[0].imshow(t_v, cmap='viridis')
                    axes[0].set_title('Target mean')
                    axes[0].axis('off')
                    axes[1].imshow(p_v, cmap='viridis')
                    axes[1].set_title('Pred mean')
                    axes[1].axis('off')
                    axes[2].imshow(d_v, cmap='inferno')
                    axes[2].set_title('|Diff|')
                    axes[2].axis('off')
                    # compute L1 on matching shapes (add batch dim to be safe)
                    try:
                        l1_vis = float(F.l1_loss(pred_cpu[i:i+1], tgt_cpu[i:i+1]))
                    except Exception:
                        # fallback: compute mean absolute difference on flattened arrays
                        l1_vis = float((pred_cpu[i] - tgt_cpu[i]).abs().mean())
                    plt.suptitle(f'epoch{epoch+1}_sample{i+1} | L1={l1_vis:.6f}')
                    out_path = os.path.join(vis_dir, f'epoch{epoch+1}_sample{i+1}.png')
                    plt.tight_layout()
                    fig.savefig(out_path, dpi=150)
                    plt.close(fig)
                print(f"Saved {n_vis} visualizations to {vis_dir}")
            except Exception as e:
                print('[warning] Visualization failed:', str(e))

        print('Training finished')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default=None)
    p.add_argument('--device', default=None)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--save-dir', type=str, default='outputs/decoder')
    p.add_argument('--channels', type=int, default=128)
    p.add_argument('--groups', type=int, default=32)
    p.add_argument('--mapper-base', type=int, default=256, help='base channel dimension for mapper')
    p.add_argument('--log-interval', type=int, default=10)
    p.add_argument('--max-iters', type=int, default=None, help='optional max total iterations across all epochs')
    p.add_argument('--max-batches', type=int, default=None)
    p.add_argument('--cosine-weight', type=float, default=0.1, help='weight for (1-cosine) term added to L1')
    p.add_argument('--save-interval', type=int, default=2, help='save checkpoint every N epochs')
    p.add_argument('--normalize-features', action='store_true', help='normalize features per-sample before loss')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.checkpoint is None:
        raise RuntimeError('Please provide --checkpoint path to pretrained backbone checkpoint')
    train(args)

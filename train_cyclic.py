import os
import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm

from scripts.common import prepare_val


class DownsampleMapper(nn.Module):
    def __init__(self, ch=128, groups=32):
        super().__init__()
        g = min(groups, ch)
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, stride=2, padding=1), nn.GroupNorm(g, ch), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, stride=2, padding=1), nn.GroupNorm(g, ch), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, stride=2, padding=1), nn.GroupNorm(g, ch), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 1)
        )

    def forward(self, z):
        return self.net(z)


class Adaptor1x1(nn.Module):
    """Small 1x1 conv adaptor to align decoder channels to neck feature channels."""
    def __init__(self, ch=128):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 1)
    def forward(self, x):
        return self.conv(x)


def move_batch_to_device(batch, device):
    for k, v in list(batch.items()):
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
        elif isinstance(v, list):
            batch[k] = [t.to(device) if isinstance(t, torch.Tensor) else t for t in v]
    return batch


def save_checkpoint(state, out_dir, epoch, prefix="cyclic_model"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fn = out_dir / f"{prefix}_epoch{epoch}.pth"
    torch.save(state, str(fn))
    return str(fn)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='/home/adeel/Gsplat_GaussianLSS/GaussianLSS/logs/GaussianLSS/2025_0806_181457/checkpoints/last.ckpt',
                   help='Path to pretrained GaussianLSS backbone checkpoint (step-1)')
    p.add_argument('--decoder-checkpoint', default='/home/adeel/Gsplat_GaussianLSS/GaussianLSS/outputs/decoder_train_epoch/decoder_backbone_final_epoch9.pth',
                   help='Path to pretrained decoder checkpoint (trained on perspective_sub)')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--alpha', type=float, default=1.0, help='weight for injected-forward loss')
    p.add_argument('--mix-alpha', type=float, default=1.0, help='mixing coefficient for mapped vs real neck features (alpha*mapped + (1-alpha)*neck). 1.0 uses mapped only.')
    p.add_argument('--save-interval', type=int, default=1, help='how often (in epochs) to save checkpoints; set to 1 to save every epoch')
    p.add_argument('--log-interval', type=int, default=200)
    p.add_argument('--save-dir', type=str, default='./outputs/cyclic_run')
    p.add_argument('--max-batches', type=int, default=None)
    p.add_argument('--calib-batches', type=int, default=10, help='how many batches to compute neck <-> decoder stats for calibration (0=disabled)')
    p.add_argument('--use-adaptor', action='store_true', help='use a small 1x1 conv adaptor (trainable) instead of fixed affine calibration')
    p.add_argument('--adaptor-lr', type=float, default=None, help='learning rate for adaptor if used (defaults to --lr)')
    p.add_argument('--only-update-on-decoder', action='store_true', help='only backpropagate using the decoder (Pass B) loss (weighted by --decoder-weight)')
    p.add_argument('--decoder-weight', type=float, default=0.2, help='weight applied to decoder loss when --only-update-on-decoder is set')
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # prepare model/network/loader — ensure perspective rendering is enabled so
    # the forward produces 'perspective_suv' (required by cyclic flow)
    overrides = ["model.enable_perspective=True"]
    model, network, loader, viz, dataset = prepare_val('GaussianLSS', device, args.checkpoint, overrides=overrides, mode='train', batch_size=args.batch_size)
    # sanity: ensure network was constructed with perspective enabled
    if not getattr(network, 'enable_perspective', False):
        raise RuntimeError('[train_cyclic] network.enable_perspective is False despite override; check Hydra config or prepare_val behavior')
    print(f"Using dataset with {len(dataset)} samples | loader batches (approx) = {len(loader)} | batch_size={args.batch_size}")

    # Load decoder and freeze
    decoder = DownsampleMapper(ch=128, groups=32).to(device)
    if args.decoder_checkpoint and os.path.exists(args.decoder_checkpoint):
        st = torch.load(args.decoder_checkpoint, map_location='cpu')
        # support state dict or wrapped dict
        if isinstance(st, dict) and 'decoder_state' in st:
            sd = st['decoder_state']
        else:
            sd = st
        decoder.load_state_dict(sd, strict=False)
        print('Loaded decoder checkpoint (strict=False)')
    decoder.eval()
    for p_ in decoder.parameters():
        p_.requires_grad = False

    # Freeze internal image backbone (feature extractor) so we only train the
    # downstream pipeline (neck, GS renderer, heads). The 'network' argument
    # returned by prepare_val is the instantiated GaussianLSS model; it owns
    # an internal `backbone` module (image encoder) and optionally a
    # `perspective_decoder` we should keep frozen as well.
    try:
        if hasattr(network, 'backbone') and network.backbone is not None:
            for p_ in network.backbone.parameters():
                p_.requires_grad = False
            print('[train_cyclic] Froze internal image backbone parameters')
    except Exception:
        pass
    try:
        if hasattr(network, 'perspective_decoder') and network.perspective_decoder is not None:
            for p_ in network.perspective_decoder.parameters():
                p_.requires_grad = False
            print('[train_cyclic] Froze internal perspective_decoder parameters')
    except Exception:
        pass
    # Ensure downstream modules (neck, gs_render, head) are trainable for Pass B
    downstream_modules = ['neck', 'gs_render', 'head']
    for mname in downstream_modules:
        try:
            if hasattr(network, mname) and getattr(network, mname) is not None:
                for p_ in getattr(network, mname).parameters():
                    p_.requires_grad = True
                print(f'[train_cyclic] Ensured downstream module {mname} is trainable')
        except Exception:
            pass

    # Optional adaptor / calibration
    adaptor = None
    if args.use_adaptor:
        adaptor = Adaptor1x1(ch=128).to(device)
        # leave adaptor trainable
        if args.adaptor_lr is None:
            adaptor_lr = args.lr
        else:
            adaptor_lr = args.adaptor_lr
    else:
        adaptor_lr = None

    # Calibration: compute per-channel mean/std of real neck and decoder mapped over a few batches
    ch_mean_real = None
    ch_std_real = None
    ch_mean_dec = None
    ch_std_dec = None
    if (args.calib_batches or 0) > 0:
        print(f"Running calibration on {args.calib_batches} batches to compute neck<->decoder stats...")
        model.eval()
        network.eval()
        cnt = 0
        valid = 0
        for batch in loader:
            if cnt >= args.calib_batches:
                break
            batch = move_batch_to_device(batch, device)
            with torch.no_grad():
                pred = network(batch)
            neck_real = pred.get('neck_features', None)
            suv = pred.get('perspective_suv', None)
            if (neck_real is None) or (suv is None):
                cnt += 1
                continue
            B, N, D, Hs, Ws = suv.shape
            suv_flat = suv.view(B * N, D, Hs, Ws)
            with torch.no_grad():
                mapped_flat = decoder(suv_flat)
            _, _, He, We = mapped_flat.shape
            mapped = mapped_flat.view(B, N, D, He, We)
            # compute per-channel stats over B,N,H,W axes
            mr = neck_real.detach().cpu().mean(axis=(0,1,3,4))
            sr = neck_real.detach().cpu().std(axis=(0,1,3,4))
            md = mapped.detach().cpu().mean(axis=(0,1,3,4))
            sd = mapped.detach().cpu().std(axis=(0,1,3,4))
            if ch_mean_real is None:
                ch_mean_real = mr
                ch_std_real = sr
                ch_mean_dec = md
                ch_std_dec = sd
            else:
                ch_mean_real += mr
                ch_std_real += sr
                ch_mean_dec += md
                ch_std_dec += sd
            cnt += 1
            valid += 1
        if valid > 0:
            ch_mean_real /= float(valid)
            ch_std_real /= float(valid)
            ch_mean_dec /= float(valid)
            ch_std_dec /= float(valid)
            print('Calibration complete. Example channels:')
            print('neck_mean[:5]=', ch_mean_real[:5].numpy())
            print('dec_mean [:5]=', ch_mean_dec[:5].numpy())
            print('neck_std [:5]=', ch_std_real[:5].numpy())
            print('dec_std  [:5]=', ch_std_dec[:5].numpy())
        else:
            ch_mean_real = None
            ch_std_real = None
            ch_mean_dec = None
            ch_std_dec = None
        # back to train
        model.train()
        network.train()

    # Optimizer: update model parameters and any learnable loss weights (model.loss_func parameters)
    # Recompute trainable params after freezing backbone/perspective_decoder above.
    # Explicitly collect params from downstream modules to avoid accidental omission
    params = []
    for name in ['neck', 'head']:
        try:
            module = getattr(network, name, None)
            if module is not None:
                params += [p for p in module.parameters() if p.requires_grad]
        except Exception:
            pass
    # Include model-level learnable parameters (e.g., loss weights)
    try:
        params += [p for p in model.loss_func.parameters() if p.requires_grad]
    except Exception:
        pass
    # Fallback: include any remaining trainable params from model (compare by id to
    # avoid elementwise tensor equality which raises on torch tensors)
    existing_ids = {id(p) for p in params}
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if id(p) in existing_ids:
            continue
        params.append(p)
        existing_ids.add(id(p))
    # if adaptor used, include its params (it should be trainable)
    if adaptor is not None:
        params += [p for p in adaptor.parameters() if p.requires_grad]
        # build param groups so adaptor can use different lr if requested
        if adaptor_lr is not None and adaptor_lr != args.lr:
            param_groups = [
                {'params': [p for p in model.parameters() if p.requires_grad], 'lr': args.lr},
                {'params': [p for p in adaptor.parameters() if p.requires_grad], 'lr': adaptor_lr}
            ]
            opt = torch.optim.Adam(param_groups)
        else:
            opt = torch.optim.Adam(params, lr=args.lr)
    else:
        opt = torch.optim.Adam(params, lr=args.lr)

    # Sanity check: ensure optimizer has params to update
    total_trainable = sum(1 for p in params if p.requires_grad)
    if total_trainable == 0:
        raise RuntimeError('No trainable parameters found: check --lr, model parameter requires_grad, or adaptor settings.')
    # Print trainable parameter count for clarity
    try:
        total_params = sum(p.numel() for p in params)
        print(f'[train_cyclic] Trainable parameter tensors: {len(params)} | total elements: {total_params}')
    except Exception:
        pass

    total_batches = len(loader)
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        running_samples = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", total=total_batches)
        it_epoch = 0
        for batch in pbar:
            # optional batch cap
            if args.max_batches is not None and it_epoch >= args.max_batches:
                break

            batch = move_batch_to_device(batch, device)

            # provide global step/epoch in batch if loss expects them
            batch['_global_step'] = int(0)
            batch['_epoch'] = int(epoch)

            # PASS A: normal forward (frozen) — do not allow gradients from Pass A
            with torch.no_grad():
                pred_a = network(batch)
                try:
                    loss_a, details_a, weights_a = model.loss_func(pred_a, batch)
                except Exception:
                    loss_a = torch.tensor(0.0, device=device)

            # PASS B: decoder -> inject neck features -> forward
            # Expectation: 'perspective_suv' is always produced by Pass A. If it's
            # missing, raise an error so the developer can correct the model/key
            # naming instead of silently falling back to other features.
            if ('perspective_suv' not in pred_a) or (pred_a['perspective_suv'] is None):
                raise RuntimeError("[train_cyclic] expected 'perspective_suv' in pred_a but it's missing or None; check model forward outputs or key naming.")
            suv = pred_a['perspective_suv']  # [B, N, D, H, W]
            B, N, D, Hs, Ws = suv.shape
            suv_flat = suv.view(B * N, D, Hs, Ws)
            with torch.no_grad():
                mapped = decoder(suv_flat)
            _, _, He, We = mapped.shape
            mapped = mapped.view(B, N, D, He, We)
            # Ensure mapped features match the neck resolution produced in Pass A
            neck_real = pred_a.get('neck_features', None)
            if neck_real is not None:
                _, _, C_neck, H_neck, W_neck = neck_real.shape
                # Channels: try to adapt if mismatch
                if mapped.shape[2] != C_neck:
                    if adaptor is not None:
                        # adaptor will be applied after interpolation
                        pass
                    else:
                        # pad or crop channels to match
                        cmap = mapped.shape[2]
                        if cmap < C_neck:
                            pad = mapped.new_zeros((B, N, C_neck - cmap, mapped.shape[3], mapped.shape[4]))
                            mapped = torch.cat([mapped, pad], dim=2)
                        elif cmap > C_neck:
                            mapped = mapped[:, :, :C_neck, :, :]
                # Spatial: resize mapped to neck spatial size if needed
                if (mapped.shape[3] != H_neck) or (mapped.shape[4] != W_neck):
                    mf = mapped.view(B * N, mapped.shape[2], mapped.shape[3], mapped.shape[4])
                    mf = torch.nn.functional.interpolate(mf, size=(H_neck, W_neck), mode='bilinear', align_corners=False)
                    mapped = mf.view(B, N, mapped.shape[2], H_neck, W_neck)
            # move to device/dtype of neck_real if present
            if neck_real is not None:
                mapped = mapped.to(neck_real.device, dtype=neck_real.dtype)
            # Apply calibration affine transform to match real neck stats if available
            if (ch_mean_real is not None) and (ch_std_real is not None):
                # mapped: [B, N, C, H, W] -> operate per-channel
                md = mapped.detach()
                # move calibration stats to device
                cm_dec = torch.from_numpy(ch_mean_dec.numpy()).to(device).view(1, 1, -1, 1, 1)
                cs_dec = torch.from_numpy(ch_std_dec.numpy()).to(device).view(1, 1, -1, 1, 1)
                cm_real = torch.from_numpy(ch_mean_real.numpy()).to(device).view(1, 1, -1, 1, 1)
                cs_real = torch.from_numpy(ch_std_real.numpy()).to(device).view(1, 1, -1, 1, 1)
                # avoid div by zero
                cs_dec = cs_dec.clamp(min=1e-6)
                # normalize decoder outputs to zero-mean unit-std using decoder stats, then scale to real neck stats
                mapped = (mapped - cm_dec) / cs_dec * cs_real + cm_real
            # Optionally pass through a small 1x1 adaptor (trainable)
            if adaptor is not None:
                # apply per (B*N, C, H, W)
                mapped_flat2 = mapped.view(B * N, D, He, We)
                mapped_flat2 = adaptor(mapped_flat2)
                mapped = mapped_flat2.view(B, N, D, He, We)

            # Mix mapped and real neck features if requested (alpha in [0,1])
            # alpha=1.0 => mapped only; alpha=0.0 => real neck only
            try:
                mix_alpha = float(args.mix_alpha)
            except Exception:
                mix_alpha = 1.0
            if (neck_real is not None) and (mix_alpha < 1.0):
                # ensure same dtype/device
                neck_real_local = neck_real.to(mapped.device, dtype=mapped.dtype)
                mapped = mix_alpha * mapped + (1.0 - mix_alpha) * neck_real_local

            # Inject at backbone-level so Pass B uses the decoder-produced dense SUV
            # features in place of backbone features[0]. The model accepts either
            # a flattened tensor [(B*N),D,H,W] or per-camera [B,N,D,H,W] under
            # 'backbone_features_injected'. We'll provide per-camera layout.
            batch['backbone_features_injected'] = mapped
            # Ensure gradients are enabled for Pass B forward: request the network
            # to preserve grads on heavy tensors (GaussianLSS.detach behavior)
            try:
                batch['_retain_grad'] = True
            except Exception:
                pass

            # Run Pass B forward and compute loss for downstream updates
            pred_b = network(batch)
            try:
                loss_b, details_b, weights_b = model.loss_func(pred_b, batch)
            except Exception:
                # ensure variables exist even if loss function fails
                loss_b = torch.tensor(0.0, device=device)
                details_b = {}
                weights_b = None

            # If loss_b ended up detached or has no grad, add a small fallback BEV-consistency loss
            try:
                needs_fallback = not (isinstance(loss_b, torch.Tensor) and getattr(loss_b, 'requires_grad', False))
            except Exception:
                needs_fallback = True
            if needs_fallback:
                try:
                    if ('bev_features' in pred_b) and ('bev_features' in pred_a):
                        bev_b = pred_b['bev_features']
                        bev_a = pred_a['bev_features']
                        if isinstance(bev_b, torch.Tensor) and isinstance(bev_a, torch.Tensor):
                            # ensure bev_a is detached and on same device/dtype
                            bev_a_local = bev_a.detach().to(bev_b.device, dtype=bev_b.dtype)
                            fallback = F.l1_loss(bev_b, bev_a_local)
                            # small weight so it doesn't overpower main losses
                            loss_b = loss_b + 0.1 * fallback if isinstance(loss_b, torch.Tensor) else 0.1 * fallback
                            # add to details for logging
                            if isinstance(details_b, dict):
                                details_b['fallback_bev_l1'] = fallback.detach()
                except Exception:
                    pass

            # Diagnostics: show what pred_b contains and loss components to help debugging
            try:
                print('\n[train_cyclic] pred_b keys:')
                if isinstance(pred_b, dict):
                    for k, v in pred_b.items():
                        if isinstance(v, torch.Tensor):
                            print(f"  {k}: tensor {tuple(v.shape)}, requires_grad={v.requires_grad}")
                        else:
                            print(f"  {k}: {type(v)}")
                else:
                    print('  pred_b not a dict, type=', type(pred_b))
            except Exception:
                pass
            try:
                print('\n[train_cyclic] loss_b details:')
                if isinstance(details_b, dict):
                    for k, v in details_b.items():
                        try:
                            print(f"  {k}: {float(v.detach().cpu()):.6f}")
                        except Exception:
                            print(f"  {k}: {v}")
                else:
                    print('  details_b not a dict')
            except Exception:
                pass
            # clean up the temporary flag so other codepaths aren't affected
            try:
                del batch['_retain_grad']
            except Exception:
                pass
            # (no fallback) Pass B always runs and must produce a gradful loss for downstream modules

            # Compute reported combined loss
            loss = loss_a + args.alpha * loss_b

            opt.zero_grad(set_to_none=True)
            if args.only_update_on_decoder:
                # Only update when decoder-derived loss is present and has a grad graph
                can_update = isinstance(loss_b, torch.Tensor) and loss_b.requires_grad and float(loss_b.detach().cpu().item()) != 0.0
                if not can_update:
                    print('\n[train_cyclic] Skipping optimizer step: --only-update-on-decoder set but loss_b has no grad or is zero.')
                    # still update running stats and continue the epoch loop
                    it_epoch += 1
                    running_loss += float(loss.item()) * batch.get('image', torch.zeros(1)).shape[0]
                    running_samples += batch.get('image', torch.zeros(1)).shape[0]
                    if it_epoch % args.log_interval == 0:
                        pbar.set_postfix(loss=f"{(running_loss / max(1, running_samples)):.4f}")
                    continue
                upd_loss = args.decoder_weight * loss_b
                upd_loss.backward()
                opt.step()
            else:
                # If combined loss doesn't require grad (e.g., loss_a was computed under no_grad
                # and loss_b ended up detached), avoid calling backward() on a non-grad tensor.
                if isinstance(loss, torch.Tensor) and getattr(loss, 'requires_grad', False):
                    loss.backward()
                    opt.step()
                else:
                    # Try to fall back to loss_b if it has a grad graph
                    if isinstance(loss_b, torch.Tensor) and getattr(loss_b, 'requires_grad', False) and float(loss_b.detach().cpu().item()) != 0.0:
                        print('\n[train_cyclic] combined loss has no grad; falling back to loss_b backward')
                        loss_b.backward()
                        opt.step()
                    else:
                        print('\n[train_cyclic] combined loss has no grad; skipping optimizer step for this batch')
                        # update running stats and continue
                        it_epoch += 1
                        try:
                            running_loss += float(loss.item()) * batch.get('image', torch.zeros(1)).shape[0]
                        except Exception:
                            running_loss += 0.0
                        running_samples += batch.get('image', torch.zeros(1)).shape[0]
                        if it_epoch % args.log_interval == 0:
                            pbar.set_postfix(loss=f"{(running_loss / max(1, running_samples)):.4f}")
                        continue

            it_epoch += 1
            running_loss += float(loss.item()) * batch.get('image', torch.zeros(1)).shape[0]
            running_samples += batch.get('image', torch.zeros(1)).shape[0]

            if it_epoch % args.log_interval == 0:
                pbar.set_postfix(loss=f"{(running_loss / max(1, running_samples)):.4f}")

        epoch_time = time.time() - epoch_start
        avg_loss = running_loss / max(1, running_samples)
        print(f"Epoch {epoch} finished | avg loss {avg_loss:.6f} | samples_seen {running_samples} | epoch_time {epoch_time:.1f}s")

        if (epoch % args.save_interval == 0) or (epoch == args.epochs):
            ckpt = save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'network_state_dict': network.state_dict(),
                'decoder_state_dict': decoder.state_dict() if decoder is not None else None,
                'optimizer_state_dict': opt.state_dict()
            }, args.save_dir, epoch)
            print('Saved model checkpoint to', ckpt)

    total_time = time.time() - start_time
    print('Training finished. Total time:', total_time)


if __name__ == '__main__':
    main()

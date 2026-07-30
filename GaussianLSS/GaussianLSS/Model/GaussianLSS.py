import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math
import logging
from gsplat.rendering import rasterization as gs_rasterize

# Optional rig-based camera deltas (SUB -> SUV).
# Try multiple strategies so notebooks/scripts can import seamlessly.
def _load_rig_cam_delta():
    try:
        from GaussianLSS.scripts.rig_utlis import CAM_DELTA as _CD  # type: ignore
        return _CD
    except Exception:
        pass
    try:
        # Fallback: relative import when executed from this file location
        import os, sys
        here = os.path.dirname(os.path.abspath(__file__))
        scripts_dir = os.path.join(os.path.dirname(os.path.dirname(here)), 'scripts')
        if scripts_dir not in sys.path:
            sys.path.append(scripts_dir)
        from rig_utlis import CAM_DELTA as _CD  # type: ignore
        return _CD
    except Exception:
        # Final fallback: zero deltas so code runs but no shift applied
        return {
            'CAM_FRONT':        torch.tensor([0.0, 0.0, 0.0]),
            'CAM_FRONT_LEFT':   torch.tensor([0.0, 0.0, 0.0]),
            'CAM_FRONT_RIGHT':  torch.tensor([0.0, 0.0, 0.0]),
            'CAM_BACK':         torch.tensor([0.0, 0.0, 0.0]),
            'CAM_BACK_LEFT':    torch.tensor([0.0, 0.0, 0.0]),
            'CAM_BACK_RIGHT':   torch.tensor([0.0, 0.0, 0.0]),
        }

RIG_CAM_DELTA = _load_rig_cam_delta()

class Normalize(nn.Module):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super().__init__()

        self.register_buffer('mean', torch.tensor(mean)[None, :, None, None], persistent=False)
        self.register_buffer('std', torch.tensor(std)[None, :, None, None], persistent=False)

    def forward(self, x):
        return (x - self.mean) / self.std

class GaussianLSS(nn.Module):
    def __init__(
            self,
            embed_dims,
            backbone,
            head,
            neck,
            decoder=nn.Identity(),
            error_tolerance=1.0,
            depth_num=64,
            opacity_filter=0.05,
            img_h=224,
            img_w=480,
            depth_start=1,
            depth_max=61,
            enable_perspective=False,
            # Performance knobs for perspective path
            persp_scale: float = 1.0,
            persp_cam_sample_k: int = 0,
            # Optional lightweight mapper for perspective features -> neck resolution
            perspective_decoder: nn.Module = None,
    ):
        super().__init__()
        
        self.norm = Normalize()
        self.backbone = backbone
        self.head = head
        self.neck = neck
        self.decoder = decoder
        self.perspective_decoder = perspective_decoder

        self.depth_num = depth_num
        self.depth_start = depth_start
        self.depth_max = depth_max
        self.gs_render = GaussianRenderer(embed_dims, opacity_filter)
        self.enable_perspective = enable_perspective
        if self.enable_perspective:
            self.persp_render = PerspectiveRenderer(embed_dims, opacity_filter)
        # perspective rendering controls
        self.persp_scale = float(persp_scale)
        self.persp_cam_sample_k = int(persp_cam_sample_k or 0)

        self.error_tolerance = error_tolerance
        self.img_h = img_h
        self.img_w = img_w
        
        bins = self.init_bin_centers()
        self.register_buffer('bins', bins, persistent=False)

    def init_bin_centers(self):
        """
        depth: b d h w
        """
        depth_range = self.depth_max - self.depth_start
        interval = depth_range / self.depth_num
        interval = interval * torch.ones((self.depth_num+1))
        interval[0] = self.depth_start
        bin_edges = torch.cumsum(interval, 0)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        return bin_centers
    
    def pred_depth(self, lidar2img, depth, coords_3d=None):
        # b, n, c, h, w = depth.shape
        if coords_3d is None:
            # bins = self.bins * self.bin_scale + self.bin_bias
            coords_3d, coords_d = get_pixel_coords_3d(self.bins, depth, lidar2img, depth_num=self.depth_num, depth_start=self.depth_start, depth_max=self.depth_max, img_h=self.img_h, img_w=self.img_w) # b n w h d 3
            coords_3d = rearrange(coords_3d, 'b n w h d c -> (b n) d h w c')
            
        depth_prob = depth.softmax(1) # (b n) depth h w
        pred_coords_3d = (depth_prob.unsqueeze(-1) * coords_3d).sum(1)  # (b n) h w 3
        
        delta_3d = pred_coords_3d.unsqueeze(1) - coords_3d
        cov = (depth_prob.unsqueeze(-1).unsqueeze(-1) * (delta_3d.unsqueeze(-1) @ delta_3d.unsqueeze(-2))).sum(1)
        scale = (self.error_tolerance ** 2) / 9 
        cov = cov * scale

        return pred_coords_3d, cov

    def forward(self, batch):
        b, n, _, _, _ = batch['image'].shape
        image = batch['image'].flatten(0, 1).contiguous()  # b n c h w

        # Some dataset modes don't precompute lidar2img; compose it if missing
        if 'lidar2img' in batch:
            lidar2img = batch['lidar2img']
        elif ('intrinsics' in batch) and ('extrinsics' in batch):
            Ks = batch['intrinsics']  # [B,N,3,3]
            Es = batch['extrinsics']  # [B,N,4,4]
            # Compose viewpad(K) @ E per camera
            B_im, N_im = Ks.shape[:2]
            device = image.device
            Ks = Ks.to(device).float()
            Es = Es.to(device).float()
            viewpads = torch.eye(4, device=device, dtype=torch.float32).view(1, 1, 4, 4).repeat(B_im, N_im, 1, 1)
            viewpads[:, :, :3, :3] = Ks
            lidar2img = viewpads @ Es
        else:
            raise KeyError("Batch missing 'lidar2img' and (intrinsics, extrinsics); cannot proceed.")
        # Run image backbone. Many backbones return a sequence/tuple of
        # multi-scale tensors; use the first element as the primary feature
        # map for downstream BEV processing. If a caller supplies
        # 'backbone_features_injected' we replace that primary tensor so
        # Pass B can consume decoder-produced features.
        features_raw = self.backbone(self.norm(image))
        # Prefer the first element when backbone returns a list/tuple
        if isinstance(features_raw, (list, tuple)) and len(features_raw) > 0:
            features_primary = features_raw[0]
        else:
            features_primary = features_raw

        # Optional injection: replace the primary backbone feature tensor.
        # Support either 'backbone_features_injected' (legacy) or
        # 'perspective_decoded' (convenient name when decoder output is used).
        try:
            injected = None
            if isinstance(batch, dict):
                if ('perspective_decoded' in batch) and (batch['perspective_decoded'] is not None):
                    injected = batch['perspective_decoded']
                elif ('backbone_features_injected' in batch) and (batch['backbone_features_injected'] is not None):
                    injected = batch['backbone_features_injected']
            if isinstance(injected, torch.Tensor):
                # prefer injected tensor
                pass
            else:
                injected = None
            if injected is not None:
                injected = injected.to(features_primary.device, dtype=features_primary.dtype)
                if isinstance(injected, torch.Tensor):
                    injected = injected.to(features_primary.device, dtype=features_primary.dtype)
                    # Accept per-camera [B,N,D,H,W] or flattened [(B*N),D,H,W]
                    if injected.ndim == 5 and injected.shape[0] == b and injected.shape[1] == n:
                        try:
                            features_primary = rearrange(injected, 'b n d h w -> (b n) d h w', b=b, n=n)
                            self._debug_backbone_features = injected
                        except Exception:
                            pass
                    elif injected.ndim == 4 and injected.shape[0] == features_primary.shape[0] and injected.shape[1] == features_primary.shape[1]:
                        features_primary = injected
                        try:
                            self._debug_backbone_features = rearrange(features_primary, '(b n) d h w -> b n d h w', b=b, n=n)
                        except Exception:
                            pass
        except Exception:
            pass

        # If backbone originally returned multiple elements, keep them but
        # replace the primary element with the possibly-injected primary
        # tensor so downstream neck receives the expected multi-scale list.
        if isinstance(features_raw, (list, tuple)) and len(features_raw) > 0:
            features_list = list(features_raw)
            features_list[0] = features_primary
            features_in = features_list
        else:
            features_in = features_primary
        # ─── debug: capture backbone features for interactive inspection and log a short summary ───
        try:
            logger = logging.getLogger(__name__)
            self._debug_backbone_features = None
            # features_primary is (B*N, C, H, W) — expose per-camera layout when possible
            try:
                self._debug_backbone_features = rearrange(features_primary, '(b n) d h w -> b n d h w', b=b, n=n)
            except Exception:
                try:
                    self._debug_backbone_features = features_primary
                except Exception:
                    self._debug_backbone_features = None
            if logger.isEnabledFor(logging.DEBUG) and (self._debug_backbone_features is not None):
                f = self._debug_backbone_features
                if isinstance(f, torch.Tensor):
                    mn = float(f.min())
                    mean = float(f.mean())
                    mx = float(f.max())
                    logger.debug(f"backbone primary shape={f.shape} min={mn:.6f} mean={mean:.6f} max={mx:.6f}")
        except Exception as _err:
            logging.getLogger(__name__).debug(f"error inspecting backbone features: {_err}")
        # ───────────────────────────────────────────────────────────────────────────────

        features, depth, opacities = self.neck(features_in)

        # NOTE: removed support for 'neck_features_injected' to keep the forward
        # function focused: we now replace backbone primary features via
        # 'backbone_features_injected' (above) and derive neck features from
        # the (possibly injected) primary tensor.

        means3D, cov3D = self.pred_depth(lidar2img, depth)
        cov3D = cov3D.flatten(-2, -1)
        cov3D = torch.cat((cov3D[..., 0:3], cov3D[..., 4:6], cov3D[..., 8:9]), dim=-1)

        features = rearrange(features, '(b n) d h w -> b (n h w) d', b=b, n=n)
        means3D = rearrange(means3D, '(b n) h w d-> b (n h w) d', b=b, n=n)
        cov3D = rearrange(cov3D, '(b n) h w d -> b (n h w) d', b=b, n=n)
        opacities = rearrange(opacities, '(b n) d h w -> b (n h w) d', b=b, n=n)

        # Determine if we should collect BEV meta (needed for tracking)
        track_k = 0
        if isinstance(batch, dict):
            track_k = int(batch.get('track_k', 0) or 0)
            if batch.get('track_subset', False) and track_k == 0:
                track_k = 10
        need_bev_meta = bool(track_k > 0 or batch.get('debug_forward', False)) if isinstance(batch, dict) else False

        bev_meta = None
        if need_bev_meta:
            x, num_gaussians, bev_meta = self.gs_render(features, means3D, cov3D, opacities, return_meta=True)
        else:
            x, num_gaussians = self.gs_render(features, means3D, cov3D, opacities, return_meta=False)
        x_bev = x  # keep a copy for outputs
        y = self.decoder(x)
        output = self.head(y)
        output['num_gaussians'] = num_gaussians
        # expose BEV rasterized features (D,H,W) per batch
        # By default, detach heavy tensors to avoid accidental graph retention in eval flows.
        # If caller sets batch['_retain_grad']=True we preserve gradients so downstream
        # losses (Pass B) can backprop through these tensors.
        detach_outputs = True
        try:
            if isinstance(batch, dict) and bool(batch.get('_retain_grad', False)):
                detach_outputs = False
        except Exception:
            pass

        output['bev_features'] = x_bev.detach() if detach_outputs else x_bev
        # Debug: expose Gaussians for downstream calibration / sparsity checks
        output['means3D'] = means3D.detach() if detach_outputs else means3D
        output['cov3D'] = cov3D.detach() if detach_outputs else cov3D
        output['opacities'] = opacities.detach() if detach_outputs else opacities
        output['features'] = features.detach() if detach_outputs else features
        # Note: removed exposing 'neck_features' per upstream simplification.

        # Optional: render the same Gaussians from real pinhole cameras if intrinsics/extrinsics are provided
        if self.enable_perspective and (('intrinsics' in batch) or ('intrinsics_raw' in batch)) and (('extrinsics' in batch) or ('extrinsics_raw' in batch)):
            # Prefer raw, non‑BEV‑augmented parameters if provided by the dataloader
            Ks = batch['intrinsics_raw'] if 'intrinsics_raw' in batch else batch['intrinsics']
            Es = batch['extrinsics_raw'] if 'extrinsics_raw' in batch else batch['extrinsics']
            # Always produce meta for perspective when enabled so point-tracking loss can use it
            return_meta = True
            # Optional: downsample perspective raster resolution to save time
            if self.persp_scale != 1.0:
                Hp = max(32, int(round(self.img_h * self.persp_scale)))
                Wp = max(32, int(round(self.img_w * self.persp_scale)))
            else:
                Hp, Wp = self.img_h, self.img_w
            # Optional: render only a random subset of cameras per step
            Bk, Nk = Ks.shape[:2]
            cam_idx = None
            if 0 < self.persp_cam_sample_k < Nk:
                # choose different subset per step to cover all over time
                perm = torch.randperm(Nk, device=Ks.device)
                sel = perm[: self.persp_cam_sample_k]
                cam_idx = sel.sort().values
                Ks_use = Ks.index_select(1, cam_idx)
                Es_use = Es.index_select(1, cam_idx)
            else:
                Ks_use, Es_use = Ks, Es
            # Expose sampled camera indices for downstream alignment
            if cam_idx is not None:
                try:
                    output['perspective_cam_idx'] = cam_idx.detach()
                except Exception:
                    output['perspective_cam_idx'] = cam_idx
            if return_meta:
                try:
                    persp, persp_meta = self.persp_render(features, means3D, cov3D, opacities, Ks_use, Es_use, Hp, Wp, return_meta=True)
                    output['perspective_meta'] = persp_meta
                    output['perspective_sub_meta'] = persp_meta  # alias
                    # expose extrinsics actually used (SUB)
                    output['extrinsics_sub'] = (Es_use.detach().clone() if isinstance(Es_use, torch.Tensor) else Es_use)
                except Exception as _err:
                    logging.getLogger(__name__).warning(
                        f"perspective (SUB) rasterization failed: {_err}. "
                        "This often happens when gsplat is built for CUDA and you run on CPU. "
                        "Falling back to a zero tensor for 'perspective'/'perspective_sub' so forward returns deterministically. "
                        "To produce real perspective renders, run on a CUDA device with a gsplat CUDA build."
                    )
                    # Create a deterministic zero tensor matching expected shape [B, N, D, Hp, Wp]
                    B_feat = features.shape[0]
                    N_use = Ks_use.shape[1]
                    D_feat = features.shape[2]
                    persp = torch.zeros((B_feat, N_use, D_feat, Hp, Wp), device=features.device, dtype=features.dtype)
                    output['perspective_meta'] = [ [{} for _ in range(N_use)] for _ in range(B_feat) ]
                    output['perspective_sub_meta'] = output['perspective_meta']
                    output['extrinsics_sub'] = (Es_use.detach().clone() if isinstance(Es_use, torch.Tensor) else Es_use)
            else:
                try:
                    persp = self.persp_render(features, means3D, cov3D, opacities, Ks_use, Es_use, Hp, Wp, return_meta=False)
                except Exception as _err:
                    logging.getLogger(__name__).warning(
                        f"perspective (SUB) rasterization failed: {_err}. Falling back to zeros. "
                        "Run on CUDA with a gsplat CUDA build to get real renders."
                    )
                    B_feat = features.shape[0]
                    N_use = Ks_use.shape[1]
                    D_feat = features.shape[2]
                    persp = torch.zeros((B_feat, N_use, D_feat, Hp, Wp), device=features.device, dtype=features.dtype)
            # shape: [B, N, D, H, W]
            output['perspective'] = persp  # backward compat
            output['perspective_sub'] = persp  # explicit name

            # New: render a translated rig variant (SUV) without rotation changes
            # We translate each camera's extrinsics by a fixed delta in ego/rig frame.
            # Expect either batch['cam_names'] or infer a 6-cam default naming order.
            # If names are missing or length mismatch, we fall back to zero deltas.
            cam_names = batch.get('cam_names', None) if isinstance(batch, dict) else None
            if cam_names is None:
                # Use a consistent nuScenes order matching common loaders
                cam_names = ['CAM_FRONT_LEFT','CAM_FRONT','CAM_FRONT_RIGHT','CAM_BACK_LEFT','CAM_BACK','CAM_BACK_RIGHT']
            # Build translated extrinsics Es_suv with same shape as Es: [B, N, 4, 4]
            try:
                Es_suv = Es.clone()
            except Exception:
                Es_suv = Es
            B_es, N_es = Es_suv.shape[:2]
            # For safety, clamp name list length to N_es
            if isinstance(cam_names, (list, tuple)) and len(cam_names) >= N_es:
                names_use = list(cam_names)[:N_es]
            else:
                names_use = [str(i) for i in range(N_es)]

            # Apply per-cam translation to t component (no rotation change)
            for n in range(N_es):
                name = str(names_use[n])
                delta = RIG_CAM_DELTA.get(name, torch.zeros(3))
                if not isinstance(delta, torch.Tensor):
                    delta = torch.tensor(delta)
                delta = delta.to(Es_suv.device, dtype=Es_suv.dtype)
                # Physically correct camera shift: for X_cam = R X + t, moving camera by δ in world → t' = t - R·δ
                Rn = Es_suv[:, n, :3, :3]  # [B,3,3]
                t_old = Es_suv[:, n, :3, 3]  # [B,3]
                t_new = t_old - (Rn @ delta)  # [B,3]
                Es_suv[:, n, :3, 3] = t_new

            Es_suv_use = Es_suv.index_select(1, cam_idx) if cam_idx is not None else Es_suv
            if return_meta:
                try:
                    persp_suv, persp_suv_meta = self.persp_render(features, means3D, cov3D, opacities, Ks_use, Es_suv_use, Hp, Wp, return_meta=True)
                    output['perspective_suv_meta'] = persp_suv_meta
                    # expose extrinsics actually used (SUV)
                    output['extrinsics_suv'] = (Es_suv.detach().clone() if isinstance(Es_suv, torch.Tensor) else Es_suv)
                except Exception as _err:
                    logging.getLogger(__name__).warning(
                        f"perspective (SUV) rasterization failed: {_err}. Falling back to zeros. "
                        "To produce real SUV renders, run on CUDA with a gsplat CUDA build."
                    )
                    B_feat = features.shape[0]
                    N_use = Es_suv_use.shape[1]
                    D_feat = features.shape[2]
                    persp_suv = torch.zeros((B_feat, N_use, D_feat, Hp, Wp), device=features.device, dtype=features.dtype)
                    output['perspective_suv_meta'] = [ [{} for _ in range(N_use)] for _ in range(B_feat) ]
                    output['extrinsics_suv'] = (Es_suv.detach().clone() if isinstance(Es_suv, torch.Tensor) else Es_suv)
            else:
                try:
                    persp_suv = self.persp_render(features, means3D, cov3D, opacities, Ks_use, Es_suv_use, Hp, Wp, return_meta=False)
                except Exception as _err:
                    logging.getLogger(__name__).warning(
                        f"perspective (SUV) rasterization failed: {_err}. Falling back to zeros. "
                        "To produce real SUV renders, run on CUDA with a gsplat CUDA build."
                    )
                    B_feat = features.shape[0]
                    N_use = Es_suv_use.shape[1]
                    D_feat = features.shape[2]
                    persp_suv = torch.zeros((B_feat, N_use, D_feat, Hp, Wp), device=features.device, dtype=features.dtype)
            # Ensure tensors are not inadvertently aliased
            output['perspective_suv'] = persp_suv.clone()
            # Optionally decode perspective_sub into neck resolution
            if self.perspective_decoder is not None:
                try:
                    src = output.get('perspective_sub', None)
                    if src is None:
                        src = output.get('perspective', None)
                    if src is not None:
                        Bp, Np, Dp, Hp, Wp = src.shape
                        dec_in = src.reshape(Bp * Np, Dp, Hp, Wp)
                        dec_out = self.perspective_decoder(dec_in)
                        dec_out = dec_out.view(Bp, Np, dec_out.shape[1], dec_out.shape[2], dec_out.shape[3])
                        output['perspective_decoded'] = dec_out
                except Exception as e:
                    logging.getLogger(__name__).warning(f"perspective_decoder failed: {e}")
            # Optional: produce a cross-view subset tracking dict
            if track_k > 0 and (bev_meta is not None) and (('perspective_meta' in output) or ('perspective_suv_meta' in output)):
                B = features.shape[0]
                H_bev, W_bev = 200, 200
                H_p, W_p = self.img_h, self.img_w
                cam_idx = int(batch.get('cam_idx', 0)) if isinstance(batch, dict) else 0
                # Choose which perspective meta to use for tracking
                track_variant = str(batch.get('track_variant', 'sub')) if isinstance(batch, dict) else 'sub'
                if track_variant.lower() == 'suv' and ('perspective_suv_meta' in output):
                    persp_meta_src = output['perspective_suv_meta']
                else:
                    persp_meta_src = output.get('perspective_meta', output.get('perspective_suv_meta'))
                tracking = []
                for i in range(B):
                    bev_m = bev_meta[i].get('means2d', None)
                    if bev_m is None:
                        tracking.append({
                            "indices": torch.empty(0, dtype=torch.long),
                            "bev_uv": torch.empty(0, 2),
                            "persp_uv": torch.empty(0, 2),
                            "cam_idx": cam_idx,
                        })
                        continue
                    persp_m_list = persp_meta_src[i]
                    if not (isinstance(persp_m_list, (list, tuple)) and len(persp_m_list) > 0):
                        tracking.append({
                            "indices": torch.empty(0, dtype=torch.long),
                            "bev_uv": torch.empty(0, 2),
                            "persp_uv": torch.empty(0, 2),
                            "cam_idx": cam_idx,
                        })
                        continue
                    ci = max(0, min(cam_idx, len(persp_m_list) - 1))
                    persp_m = persp_m_list[ci].get('means2d', None)
                    if persp_m is None:
                        tracking.append({
                            "indices": torch.empty(0, dtype=torch.long),
                            "bev_uv": torch.empty(0, 2),
                            "persp_uv": torch.empty(0, 2),
                            "cam_idx": ci,
                        })
                        continue
                    # Coerce to [G,2]
                    def _to_uv(t):
                        if not isinstance(t, torch.Tensor):
                            return None
                        x = t
                        # try to squeeze batch/cam dims
                        while x.ndim > 2 and x.shape[-1] != 2:
                            # squeeze leading singleton dims if any
                            if x.shape[0] == 1:
                                x = x.squeeze(0)
                            elif x.shape[0] > 1 and x.shape[1] == 1:
                                x = x.squeeze(1)
                            else:
                                break
                        if x.ndim == 1:
                            # if we somehow received a single value per Gaussian, tile to (2,) to avoid index errors
                            x = x.unsqueeze(-1)
                        if x.ndim >= 2:
                            if x.shape[-1] == 1:
                                x = x.repeat_interleave(2, dim=-1)
                            if x.shape[-1] >= 2:
                                return x.reshape(-1, x.shape[-1])[:, :2]
                        return None
                    bev_uv = _to_uv(bev_m)
                    persp_uv = _to_uv(persp_m)
                    if bev_uv is None or persp_uv is None:
                        tracking.append({
                            "indices": torch.empty(0, dtype=torch.long),
                            "bev_uv": torch.empty(0, 2),
                            "persp_uv": torch.empty(0, 2),
                            "cam_idx": ci,
                        })
                        continue
                    bev_mask = (bev_uv[:, 0] >= 0) & (bev_uv[:, 0] < W_bev) & (bev_uv[:, 1] >= 0) & (bev_uv[:, 1] < H_bev)
                    persp_mask = (persp_uv[:, 0] >= 0) & (persp_uv[:, 0] < W_p) & (persp_uv[:, 1] >= 0) & (persp_uv[:, 1] < H_p)
                    both = bev_mask & persp_mask
                    idx_all = torch.nonzero(both, as_tuple=False).squeeze(1)
                    if idx_all.numel() == 0:
                        Gc = bev_uv.shape[0]
                        take = min(track_k, Gc)
                        sel = torch.randperm(Gc)[:take]
                    else:
                        take = min(track_k, idx_all.numel())
                        sel = idx_all[torch.randperm(idx_all.numel())[:take]]
                    tracking.append({
                        "indices": sel.clone(),
                        "bev_uv": bev_uv[sel].clone(),
                        "persp_uv": persp_uv[sel].clone(),
                        "cam_idx": ci,
                    })
                output['track'] = tracking
        return output
    
class BEVCamera:
    def __init__(self, x_range=(-50, 50), y_range=(-50, 50), image_size=200):
        # Orthographic projection parameters
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.image_width = image_size
        self.image_height = image_size

        # Set up FoV to cover the range [-50, 50] for both X and Y
        self.FoVx = (self.x_max - self.x_min)  # Width of the scene in world coordinates
        self.FoVy = (self.y_max - self.y_min)  # Height of the scene in world coordinates

        # Camera position: placed above the scene, looking down along Z-axis
        self.camera_center = torch.tensor([0, 0, 0], dtype=torch.float32)  # High above Z-axis

        # Orthographic projection matrix for BEV
        self.set_transform()
    
    def set_transform(self, h=200, w=200, h_meters=100, w_meters=100):
        """ Set up an orthographic projection matrix for BEV. """
        # Create an orthographic projection matrix
        sh = h / h_meters
        sw = w / w_meters
        self.world_view_transform = torch.tensor([
            [ 0.,  sh,  0.,         0.],
            [ sw,  0.,  0.,         0.],
            [ 0.,  0.,  0.,         0.],
            [ 0.,  0.,  0.,         0.],
        ], dtype=torch.float32)

        self.full_proj_transform = torch.tensor([
            [ 0., -sh,  0.,          h/2.],
            [-sw,   0.,  0.,         w/2.],
            [ 0.,  0.,  0.,           1.],
            [ 0.,  0.,  0.,           1.],
        ], dtype=torch.float32)

    def set_size(self, h, w):
        self.image_height = h
        self.image_width = w

class GaussianRenderer(nn.Module):
    def __init__(self, embed_dims, threshold=0.05):
        super().__init__()
        self.viewpoint_camera = BEVCamera()
        self.embed_dims = embed_dims
        self.threshold = threshold

    def forward(self, features, means3D, cov3D, opacities, return_meta: bool = False):
        """
        features: b G d
        means3D : b G 3
        cov3D   : b G 6  (upper‑tri packed: xx,xy,xz,yy,yz,zz)
        opacities: b G 1
        """
        b = features.shape[0]
        device = means3D.device

        # prepare BEV orthographic camera once
        H = W = 200

        # pixel‑per‑metre scaling so that x ∈ [x_min,x_max] → u ∈ [0,W]
        x_extent = self.viewpoint_camera.x_max - self.viewpoint_camera.x_min  # ≈102.4 m
        y_extent = self.viewpoint_camera.y_max - self.viewpoint_camera.y_min  # ≈102.4 m
        fx = W / x_extent
        fy = H / y_extent
        cx, cy = W * 0.5, H * 0.5

        K = torch.tensor([[fx, 0., cx],
                          [0., fy, cy],
                          [0., 0.,  1.]], device=device)[None, None]  # [1,1,3,3]

        # rotation: swap X ↔ Y and flip both axes (180° rotation) so BEV is upright
        viewmat = torch.eye(4, device=device)
        viewmat[0, 0] = 0.0
        viewmat[0, 1] = -1.0   # world‑y → image‑x (flip)
        viewmat[1, 0] = -1.0   # world‑x → image‑y (flip)
        viewmat[1, 1] = 0.0
        viewmat[2, 2] = -1.0  # look down −Z
        viewmat[2, 3] = 10.0  # 10 m above ground
        viewmats = viewmat[None, None]                                   # [1,1,4,4]

        bg_color = torch.zeros(self.embed_dims, device=device)[None, None]  # [1,1,D]

        bev_out = []
        metas_all = []
        mask = (opacities[..., 0] > self.threshold)  # [b,G]
        for i in range(b):
            # diff‑gaussian checkpoints store the lower‑triangular Cholesky factor L
            L = torch.zeros((mask.shape[1], 3, 3), device=device, dtype=torch.float32)
            tril_idx = ([0, 1, 1, 2, 2, 2], [0, 0, 1, 0, 1, 2])
            L[:, tril_idx[0], tril_idx[1]] = cov3D[i].to(torch.float32)
            Sigma = L @ L.transpose(-1, -2)

            # Ensure full precision for gsplat kernels under AMP
            from torch.cuda.amp import autocast
            with autocast(enabled=False):
                render, _, meta = gs_rasterize(
                    means=means3D[i][None].to(torch.float32),
                    quats=None,
                    scales=None,
                    covars=Sigma[None].to(torch.float32),
                    opacities=opacities[i][None, :, 0].to(torch.float32),
                    colors=features[i][None].to(torch.float32),
                    viewmats=viewmats.to(torch.float32),
                    Ks=K.to(torch.float32),
                    width=W,
                    height=H,
                    backgrounds=bg_color.to(torch.float32),
                    camera_model="ortho",
                    render_mode="RGB",
                    packed=False,
                )

            # Optional internal debug metadata collection removed to avoid
            # noisy prints during training/inference. If needed, enable
            # proper logging via a logger at debug level.

            rendered = render.squeeze(0).squeeze(0).permute(2, 0, 1)
            bev_out.append(rendered)
            if return_meta:
                # store lightweight, squeezed meta for tracking
                def _sq(t):
                    if isinstance(t, torch.Tensor) and t.ndim >= 2 and t.shape[-1] == 2:
                        return t.squeeze().reshape(-1, 2).detach().cpu()
                    return t.detach().cpu() if isinstance(t, torch.Tensor) else t
                meta_keep = {
                    k: _sq(v)
                    for k, v in meta.items()
                    if k in ("radii", "means2d", "contribs", "geom_mask")
                }
                metas_all.append(meta_keep)

        x = torch.stack(bev_out, dim=0)
        num_gaussians = mask.float().sum(1).mean().cpu()
        if return_meta:
            return x, num_gaussians, metas_all
        return x, num_gaussians
        
    # set_Rasterizer and set_render_scale methods removed

class PerspectiveRenderer(nn.Module):
    """Pinhole renderer for per-camera views using gsplat.

    Inputs:
      - features:   [B, G, D]
      - means3D:    [B, G, 3]   (in LiDAR/world coords used by training)
      - cov3D:      [B, G, 6]   (packed lower/upper triangular; we reconstruct PD cov)
      - opacities:  [B, G, 1]
      - Ks:         [B, N, 3, 3]
      - Es:         [B, N, 4, 4] (world(lidar)->cam)
    Returns:
      - rendered:   [B, N, D, H, W]
    """
    def __init__(self, embed_dims, threshold=0.05, cov_inflate: float = 1.0):
        super().__init__()
        self.embed_dims = embed_dims
        self.threshold = threshold
        # Multiply world covariance by this factor^2 to adjust footprint (diagnostic)
        self.cov_inflate = cov_inflate

    def forward(self, features, means3D, cov3D, opacities, Ks, Es, H, W, return_meta: bool = False):
        b, G, D = features.shape
        device = features.device

        # background color in D-dim feature space
        bg = torch.zeros(self.embed_dims, device=device, dtype=torch.float32)[None, None]  # [1,1,D]

        outputs = []
        metas_all = []  # collect per‑batch, per‑cam rasterizer meta if requested
        for i in range(b):
            # Reconstruct world covariance Σ using the same scheme as BEV path
            # Treat the 6 packed values as lower-triangular entries of a Cholesky-like L
            # L indices (row,col): (0,0)=xx, (1,0)=xy, (1,1)=yy, (2,0)=xz, (2,1)=yz, (2,2)=zz
            L = torch.zeros((G, 3, 3), device=device, dtype=torch.float32)
            tril_idx = ([0, 1, 1, 2, 2, 2], [0, 0, 1, 0, 1, 2])
            L[:, tril_idx[0], tril_idx[1]] = cov3D[i].to(torch.float32)
            Sigma = L @ L.transpose(-1, -2)  # PSD
            if self.cov_inflate != 1.0:
                Sigma = Sigma * (self.cov_inflate ** 2)

            cams_out = []
            cams_meta = []
            for n in range(Ks.shape[1]):
                K = Ks[i, n][None, None]                # [1,1,3,3]
                viewmats = Es[i, n][None, None]         # [1,1,4,4]

                # Ensure full precision for gsplat kernels under AMP
                from torch.cuda.amp import autocast
                with autocast(enabled=False):
                    render, _alphas, _meta = gs_rasterize(
                        means       = means3D[i][None].to(torch.float32),           # [1,G,3]
                        quats       = None,
                        scales      = None,
                        covars      = Sigma[None].to(torch.float32),                # [1,G,3,3]
                        opacities   = opacities[i][None, :, 0].to(torch.float32),   # [1,G]
                        colors      = features[i][None].to(torch.float32),          # [1,G,D]
                        viewmats    = viewmats.to(torch.float32),                   # [1,1,4,4]
                        Ks          = K.to(torch.float32),                          # [1,1,3,3]
                        width       = W, height = H,
                        backgrounds = bg.to(torch.float32),                         # [1,1,D]
                        camera_model= "pinhole",
                        eps2d=1.5,                 # ↑ try 0.5–1.5 to get a visible min radius
                        near_plane=0.1, far_plane=120.0,
                        render_mode = "RGB",
                        packed      = False,
                    )

                img = render.squeeze(0).squeeze(0).permute(2, 0, 1)  # D,H,W
                cams_out.append(img)
                if return_meta:
                    # keep only lightweight fields if present and attach per-pixel alpha coverage
                    meta_keep = {
                        k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                        for k, v in _meta.items()
                        if k in ("radii", "means2d", "contribs", "geom_mask")
                    }
                    try:
                        # _alphas shape is [1,1,H,W,1] or [1,1,H,W]; squeeze to [H,W]
                        a = _alphas
                        while hasattr(a, 'ndim') and a.ndim > 2:
                            if a.shape[0] == 1:
                                a = a.squeeze(0)
                            elif a.shape[0] > 1:
                                break
                            if a.ndim > 2 and a.shape[0] == 1:
                                a = a.squeeze(0)
                            if a.ndim > 2 and a.shape[-1] == 1:
                                a = a.squeeze(-1)
                        if isinstance(a, torch.Tensor) and a.ndim == 2:
                            meta_keep['alpha'] = a.detach().cpu()
                    except Exception:
                        pass
                    cams_meta.append(meta_keep)

            cams_out = torch.stack(cams_out, dim=0)  # N,D,H,W
            outputs.append(cams_out)
            if return_meta:
                metas_all.append(cams_meta)

        rendered = torch.stack(outputs, dim=0)  # B,N,D,H,W
        if return_meta:
            return rendered, metas_all
        return rendered

@torch.no_grad()
def get_pixel_coords_3d(coords_d, depth, lidar2img, img_h=224, img_w=480, depth_num=64, depth_start=1, depth_max=61):
    eps = 1e-5
    
    B, N = lidar2img.shape[:2]
    H, W = depth.shape[-2:]
    scale = img_h // H
    # coords_h = torch.linspace(scale // 2, img_h - scale//2, H, device=depth.device).float()
    # coords_w = torch.linspace(scale // 2, img_w - scale//2, W, device=depth.device).float()
    coords_h = torch.linspace(0, 1, H, device=depth.device).float() * img_h
    coords_w = torch.linspace(0, 1, W, device=depth.device).float() * img_w
    # coords_d = get_bin_centers(depth_max, depth_start, depth_num).to(depth.device)
    # coords_d = coords_d * bin_scale + bin_bias

    D = coords_d.shape[0]
    coords = torch.stack(torch.meshgrid([coords_w, coords_h, coords_d], indexing='ij')).permute(1, 2, 3, 0) # W, H, D, 3
    coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)
    coords[..., :2] = coords[..., :2] * torch.maximum(coords[..., 2:3], torch.ones_like(coords[..., 2:3])*eps)
    # Ensure FP32 for matrix inverse under AMP to avoid half-precision restrictions
    img2lidars = lidar2img.to(torch.float32).inverse() # b n 4 4

    coords = coords.view(1, 1, W, H, D, 4, 1).repeat(B, N, 1, 1, 1, 1, 1)
    img2lidars = img2lidars.view(B, N, 1, 1, 1, 4, 4).repeat(1, 1, W, H, D, 1, 1)
    coords3d = torch.matmul(img2lidars, coords).squeeze(-1)[..., :3] # B N W H D 3

    return coords3d, coords_d

# @torch.no_grad()
def get_bin_centers(max_depth, min_depth, depth_num):
    """
    depth: b d h w
    """
    depth_range = max_depth - min_depth
    interval = depth_range / depth_num
    interval = interval * torch.ones((depth_num+1))
    interval[0] = min_depth
    # interval = torch.cat([torch.ones_like(depth) * min_depth, interval], 1)

    bin_edges = torch.cumsum(interval, 0)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    return bin_centers
    


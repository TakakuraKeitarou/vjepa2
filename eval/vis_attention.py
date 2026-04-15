import argparse
import os
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2

from src.datasets.data_manager import init_data
from src.masks.multiseq_multiblock3d import MaskCollator
from app.vjepa_2_1.utils import init_video_model
from app.vjepa_2_1.transforms import make_transforms
import copy

def parse_args():
    parser = argparse.ArgumentParser(description="V-JEPA 2.1 Attention Rollout Visualization")
    parser.add_argument("--fname", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_dir", type=str, default="attention_results")
    parser.add_argument("--num_samples", type=int, default=10)
    return parser.parse_args()

def strip_module_prefix(state_dict):
    new_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_dict[k[7:]] = v
        else:
            new_dict[k] = v
    return new_dict


# QKVの出力をフックして、Attention行列を手動で計算する
qkv_outputs = []
def hook_qkv(module, input, output):
    """QKV Linear層の出力を保存する"""
    qkv_outputs.append(output.detach().cpu())


def compute_attention_from_qkv(qkv_out, num_heads):
    """QKV出力からAttention行列を手動で計算する"""
    B, N, C3 = qkv_out.shape
    head_dim = C3 // (3 * num_heads)
    qkv = qkv_out.unflatten(-1, (3, num_heads, -1)).permute(2, 0, 3, 1, 4)
    q, k = qkv[0], qkv[1] # (B, heads, N, head_dim)
    scale = head_dim ** -0.5
    attn = (q @ k.transpose(-2, -1)) * scale # (B, heads, N, N)
    attn = attn.softmax(dim=-1)
    return attn


def attention_rollout(attn_list):
    """
    Attention Rollout: 残差接続を考慮して全層のAttentionを累積する。
    各層で 0.5*I + 0.5*A として残差を反映する。
    """
    result = None
    for attn in attn_list:
        # 全ヘッドの平均
        attn_heads_mean = attn.mean(dim=1) # (B, N, N)
        # 残差接続の反映: 0.5*Identity + 0.5*Attention
        I = torch.eye(attn_heads_mean.size(-1))
        attn_with_residual = 0.5 * I + 0.5 * attn_heads_mean
        # 各行を正規化
        attn_with_residual = attn_with_residual / attn_with_residual.sum(dim=-1, keepdim=True)
        if result is None:
            result = attn_with_residual
        else:
            result = attn_with_residual @ result
    return result # (B, N, N)


def visualize_sample(original_clip, rollout_map, tubelet_size, patch_size, save_path, sample_idx):
    """1サンプルのAttention Rolloutを描画"""
    C, T, H, W = original_clip.shape
    T_patches = T // tubelet_size
    H_patches = H // patch_size
    W_patches = W // patch_size
    expected_N = T_patches * H_patches * W_patches

    # rollout_map: (N,) - 各パッチへの累積注目度
    if rollout_map.shape[0] != expected_N:
        rollout_map = rollout_map[-expected_N:]
    rollout_map = rollout_map.reshape(T_patches, H_patches, W_patches)

    frame_indices = np.linspace(0, T - 1, min(4, T), dtype=int)
    n_frames = len(frame_indices)

    fig, axes = plt.subplots(2, n_frames, figsize=(4 * n_frames, 8))
    if n_frames == 1:
        axes = axes.reshape(2, 1)

    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

    for col, t_disp in enumerate(frame_indices):
        img = original_clip[:, t_disp, :, :]
        img = img * std + mean
        img = np.clip(img.transpose(1, 2, 0), 0, 1)

        t_patch_idx = min(t_disp // tubelet_size, T_patches - 1)
        heatmap = rollout_map[t_patch_idx]

        # パーセンタイルで正規化（極端な外れ値を除去）
        vmin = np.percentile(heatmap, 2)
        vmax = np.percentile(heatmap, 98)
        heatmap = np.clip((heatmap - vmin) / (vmax - vmin + 1e-8), 0, 1)

        heatmap_resized = cv2.resize(heatmap, (W, H), interpolation=cv2.INTER_CUBIC)
        heatmap_colored = plt.get_cmap("inferno")(heatmap_resized)[:, :, :3]
        overlay = img * 0.5 + heatmap_colored * 0.5

        axes[0, col].imshow(img)
        axes[0, col].set_title(f"Frame {t_disp}", fontsize=10)
        axes[0, col].axis('off')

        axes[1, col].imshow(overlay)
        axes[1, col].set_title(f"Rollout (Frame {t_disp})", fontsize=10)
        axes[1, col].axis('off')

    fig.suptitle(f"Sample #{sample_idx} - Attention Rollout", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.fname, "r") as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    
    data_cfg = params["data"]
    mask_cfg = params["mask"]
    model_cfg = params["model"]
    loss_cfg = params["loss"]

    dataset_fpcs = data_cfg["dataset_fpcs"]
    
    transform = make_transforms(
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(1.0, 1.0),
        random_resize_scale=(1.0, 1.0),
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
        crop_size=data_cfg["crop_size"],
    )

    mask_collator = MaskCollator(
        cfgs_mask=mask_cfg,
        dataset_fpcs=dataset_fpcs,
        crop_size=data_cfg["crop_size"],
        patch_size=data_cfg["patch_size"],
        tubelet_size=data_cfg.get("tubelet_size", 2),
    )

    loader, _ = init_data(
        data=data_cfg["dataset_type"],
        root_path=[args.val_csv],
        batch_size=1,
        training=False,
        dataset_fpcs=dataset_fpcs,
        fps=data_cfg.get("fps", 4),
        transform=transform,
        rank=0,
        world_size=1,
        datasets_weights=[1.0],
        collator=mask_collator,
        num_workers=2,
        pin_mem=True,
    )

    # SDPAを無効化してAttention行列を明示的に計算させる
    encoder, _ = init_video_model(
        device,
        patch_size=data_cfg["patch_size"],
        max_num_frames=max(dataset_fpcs),
        tubelet_size=data_cfg.get("tubelet_size", 2),
        model_name=model_cfg["model_name"],
        crop_size=data_cfg["crop_size"],
        pred_depth=model_cfg.get("pred_depth", 6),
        pred_embed_dim=model_cfg.get("pred_embed_dim", 384),
        pred_num_heads=model_cfg.get("pred_num_heads", None),
        use_sdpa=False,  # SDPAを無効化してAttention行列を取得可能にする
        use_rope=model_cfg.get("use_rope", False),
        uniform_power=model_cfg.get("uniform_power", False),
        interpolate_rope=model_cfg.get("interpolate_rope", False),
        modality_embedding=model_cfg.get("modality_embedding", False),
        return_all_tokens=loss_cfg.get("predict_all", False),
        use_mask_tokens=model_cfg.get("use_mask_tokens", False),
        num_mask_tokens=model_cfg.get("num_mask_tokens", 2),
        zero_init_mask_tokens=model_cfg.get("zero_init_mask_tokens", True),
        use_activation_checkpointing=False,
    )
    target_encoder = copy.deepcopy(encoder)

    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    target_encoder.load_state_dict(strip_module_prefix(checkpoint["target_encoder"]), strict=False)
    target_encoder.to(device).eval()
    del checkpoint

    # Attentionを返すフラグを立てる
    target_encoder.backbone.attn_out = True
    num_heads = target_encoder.backbone.num_heads
    num_layers = len(target_encoder.backbone.blocks)

    # 全層のQKV出力にHookを登録
    hooks = []
    for blk in target_encoder.backbone.blocks:
        h = blk.attn.qkv.register_forward_hook(hook_qkv)
        hooks.append(h)

    tubelet_size = data_cfg.get("tubelet_size", 2)
    patch_size = data_cfg["patch_size"]

    global qkv_outputs

    print(f"[*] Generating {args.num_samples} Attention Rollout Maps -> {args.output_dir}/")
    for i, sample in enumerate(loader):
        if i >= args.num_samples:
            break

        qkv_outputs.clear()

        fpc_sample = sample[0]
        udata, masks_enc, masks_pred = fpc_sample
        clips = [udata[0][0].to(device, non_blocking=True)]
        original_clip = clips[0][0].cpu().numpy()

        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
                _ = target_encoder(clips, gram_mode=False, training_mode=False)

        # 全層のAttention行列を計算
        all_attn = []
        for qkv_out in qkv_outputs:
            qkv_out = qkv_out.float()
            attn = compute_attention_from_qkv(qkv_out, num_heads)
            all_attn.append(attn)

        # Attention Rollout
        rollout = attention_rollout(all_attn)
        # rollout: (B, N, N) -> 各パッチから全体パッチへの累積注目
        # 全パッチから各パッチへの注目度の平均
        rollout_map = rollout[0].mean(dim=0).numpy() # (N,)

        save_path = os.path.join(args.output_dir, f"attention_{i:03d}.png")
        visualize_sample(original_clip, rollout_map, tubelet_size, patch_size, save_path, i)
        print(f"  [{i+1}/{args.num_samples}] Saved: {save_path}")

    for h in hooks:
        h.remove()
    print(f"[*] Done! {args.num_samples} images saved to {args.output_dir}/")

if __name__ == "__main__":
    main()

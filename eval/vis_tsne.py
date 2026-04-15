import argparse
import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from src.datasets.data_manager import init_data
from src.masks.multiseq_multiblock3d import MaskCollator
from app.vjepa_2_1.utils import init_video_model
from app.vjepa_2_1.transforms import make_transforms
import copy

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate V-JEPA 2.1 t-SNE Clustering")
    parser.add_argument("--fname", type=str, required=True, help="Path to pretraining config")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--val_csv", type=str, required=True, help="Path to validation CSV")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output", type=str, default="tsne_cluster.png")
    parser.add_argument("--max_samples", type=int, default=300, help="Max number of videos to cluster")
    return parser.parse_args()

def strip_module_prefix(state_dict):
    new_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_dict[k[7:]] = v
        else:
            new_dict[k] = v
    return new_dict

def main():
    args = parse_args()
    device = torch.device(args.device)

    with open(args.fname, "r") as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    
    data_cfg = params["data"]
    mask_cfg = params["mask"]
    model_cfg = params["model"]
    loss_cfg = params["loss"]

    dataset_fpcs = data_cfg["dataset_fpcs"]
    # t-SNE用には少し多めのバッチサイズで回す
    batch_size = 16
    
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
        batch_size=batch_size,
        training=False,
        dataset_fpcs=dataset_fpcs,
        fps=data_cfg.get("fps", 4),
        transform=transform,
        rank=0,
        world_size=1,
        datasets_weights=[1.0],
        collator=mask_collator,
        num_workers=4,
        pin_mem=True,
    )

    print("[*] Initializing Target Encoder...")
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
        use_sdpa=model_cfg.get("use_sdpa", False),
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

    print(f"[*] Loading Checkpoint: {args.ckpt}")
    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    target_encoder.load_state_dict(strip_module_prefix(checkpoint["target_encoder"]), strict=False)
    target_encoder.to(device).eval()

    features_list = []
    
    print("[*] Extracting features for t-SNE...")
    num_samples = 0

    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
            for itr, sample in enumerate(loader):
                if num_samples >= args.max_samples:
                    break
                
                all_clips = []
                for fpc_sample in sample:
                    udata, masks_enc, masks_pred = fpc_sample
                    all_clips.append(udata[0][0].to(device, non_blocking=True))

                # training_mode=False にすることで次元数を1408に抑える
                h = target_encoder(all_clips, gram_mode=False, training_mode=False)
                
                # hはリストになっていて、各要素がバッチごとの特徴量 (B, N, D)
                for h_batch in h:
                    # Global Average Pooling にて1動画を1点ベクトル (D,) へ圧縮する
                    h_gap = torch.mean(h_batch, dim=1) # (B, D)
                    features_list.append(h_gap.cpu().numpy())
                    num_samples += h_batch.shape[0]

                print(f"  Extracted {num_samples} samples...")

    features = np.concatenate(features_list, axis=0) # (Total_B, D)
    print(f"[*] Total Extracted Feature Shape: {features.shape}")

    # PCAでノイズ除去・次元圧縮を行ってからt-SNEに入力する標準プラクティス
    pca_dim = min(50, features.shape[0])
    print(f"[*] Reducing dimensions to {pca_dim} using PCA...")
    pca = PCA(n_components=pca_dim)
    features_pca = pca.fit_transform(features)

    print("[*] Running t-SNE (2D)...")
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    tsne_results = tsne.fit_transform(features_pca)

    # プロット
    print("[*] Plotting...")
    plt.figure(figsize=(10, 8))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], alpha=0.7, color='indigo', s=20, edgecolors='w', linewidth=0.5)
    plt.title('V-JEPA 2.1 Latent Space Clustering (t-SNE)', fontsize=14)
    plt.xlabel('t-SNE Dim 1')
    plt.ylabel('t-SNE Dim 2')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    plt.savefig(args.output, dpi=200)
    print(f"🎯 t-SNE Cluster Plot saved to: {args.output}")

if __name__ == "__main__":
    main()

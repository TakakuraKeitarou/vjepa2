import argparse
import os
import yaml
import torch
import numpy as np

# V-JEPA modules
from src.datasets.data_manager import init_data
from src.masks.multiseq_multiblock3d import MaskCollator
from src.masks.utils import apply_masks
from app.vjepa_2_1.utils import init_video_model, load_checkpoint
from src.utils.distributed import init_distributed

def loss_fn(z, h, masks_to_apply):
    h = [apply_masks(hi, mi, concat=False) for hi, mi in zip(h, masks_to_apply)]
    loss, n = 0, 0
    for zi, hi in zip(z, h):
        for zij, hij in zip(zi, hi):
            loss += torch.mean(torch.abs(zij - hij))
            n += 1
    loss /= n
    return loss

from app.vjepa_2_1.transforms import make_transforms

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate V-JEPA 2.1 World Model Loss")
    parser.add_argument("--fname", type=str, required=True, help="Path to pretraining config (yaml)")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint (e.g. best.pth.tar)")
    parser.add_argument("--val_csv", type=str, required=True, help="Path to validation CSV")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device)

    # 1. 設定の読み込み
    with open(args.fname, "r") as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    
    data_cfg = params["data"]
    mask_cfg = params["mask"]
    model_cfg = params["model"]

    # 2. データの初期化 (Validation Dataset)
    dataset_fpcs = data_cfg["dataset_fpcs"]
    batch_size = data_cfg["batch_size"]
    
    print(f"[*] Loading Validation Dataset: {args.val_csv}")
    
    transform = make_transforms(
        random_horizontal_flip=False, # 評価時は固定
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
        root_path=[args.val_csv], # 評価用のCSVを指定
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
        log_dir=None,
    )

    loss_cfg = params["loss"]

    # 3. モデルの初期化
    print(f"[*] Initializing Model ({model_cfg['model_name']})")
    encoder, predictor = init_video_model(
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
        use_activation_checkpointing=False, # 推論時は不要
    )
    import copy
    target_encoder = copy.deepcopy(encoder)

    # 4. 重みのロード
    print(f"[*] Loading Checkpoint: {args.ckpt}")
    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    
    # マルチGPU(DistributedDataParallel)で保存された重みから "module." を除去する関数
    def strip_module_prefix(state_dict):
        new_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_dict[k[7:]] = v
            else:
                new_dict[k] = v
        return new_dict

    encoder.load_state_dict(strip_module_prefix(checkpoint["encoder"]), strict=False)
    predictor.load_state_dict(strip_module_prefix(checkpoint["predictor"]), strict=False)
    target_encoder.load_state_dict(strip_module_prefix(checkpoint["target_encoder"]), strict=False)
    
    encoder.to(device).eval()
    predictor.to(device).eval()
    target_encoder.to(device).eval()

    # 5. 評価ループ
    print("[*] Starting Evaluation...")
    total_loss = 0.0
    num_batches = 0
    
    dtype = torch.bfloat16
    mixed_precision = True

    with torch.no_grad(): # 勾配計算を完全にオフにする（評価モード）
        for itr, sample in enumerate(loader):
            all_clips, all_masks_enc, all_masks_pred = [], [], []
            for fpc_sample in sample:
                udata, masks_enc, masks_pred = fpc_sample
                all_clips.append(udata[0][0].to(device, non_blocking=True))
                all_masks_enc.append([m.to(device, non_blocking=True) for m in masks_enc])
                all_masks_pred.append([m.to(device, non_blocking=True) for m in masks_pred])

            with torch.cuda.amp.autocast(dtype=dtype, enabled=mixed_precision):
                # Contextの特徴抽出 (training_mode=Trueで階層特徴を結合して出力)
                h = encoder(all_clips, all_masks_enc, gram_mode=False, training_mode=True)
                
                # 未来の予測
                q = []
                for i in range(len(all_clips)):
                    if len(h) == len(all_clips):
                        q.append(h[i])
                    else:
                        q.append(h[0])
                z, _ = predictor(q, all_masks_enc, all_masks_pred, mod="video")
                
                # 実際の未来の特徴抽出 (Target)
                import torch.nn.functional as F
                h_ema_raw = target_encoder(all_clips, gram_mode=False, training_mode=True)
                embed_dim_encoder = target_encoder.embed_dim
                h_ema = []
                for hi in h_ema_raw:
                    # 4つの階層特徴(各embed_dim=1408)をそれぞれLayerNormして結合
                    hi_0 = F.layer_norm(hi[:, :, :embed_dim_encoder], (embed_dim_encoder,))
                    hi_1 = F.layer_norm(hi[:, :, embed_dim_encoder : embed_dim_encoder * 2], (embed_dim_encoder,))
                    hi_2 = F.layer_norm(hi[:, :, embed_dim_encoder * 2 : embed_dim_encoder * 3], (embed_dim_encoder,))
                    hi_3 = F.layer_norm(hi[:, :, -embed_dim_encoder:], (embed_dim_encoder,))
                    hi_norm = torch.cat([hi_0, hi_1, hi_2, hi_3], dim=2)
                    h_ema.append(hi_norm)
                
                # 誤差の計算
                loss = loss_fn(z, h_ema, all_masks_pred)

            
            val_loss = float(loss.item())
            total_loss += val_loss
            num_batches += 1
            
            if itr % 10 == 0:
                print(f"  Batch {itr} / {len(loader)} - Loss: {val_loss:.4f}")

    avg_loss = total_loss / num_batches
    print("========================================")
    print(f"🎯 Validation Completed!")
    print(f"🎯 Average Prediction Loss: {avg_loss:.4f}")
    print("========================================")

if __name__ == "__main__":
    main()

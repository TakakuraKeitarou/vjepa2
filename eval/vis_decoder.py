import argparse
import sys
import os
import yaml
import torch
import torchvision

from pathlib import Path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.vjepa_2_1.utils import init_video_model
from app.vjepa_2_1.models.diffusion_decoder import PatchDiffusionDecoder
from app.vjepa_2_1.transforms import make_transforms
from src.datasets.data_manager import init_data
from src.masks.multiseq_multiblock3d import MaskCollator

def patchify(imgs, p_size, t_size):
    B, C, T, H, W = imgs.shape
    h, w, l = H // p_size, W // p_size, max(1, T // t_size)
    t_size = min(T, t_size)
    x = imgs.reshape(B, C, l, t_size, h, p_size, w, p_size)
    x = x.permute(0, 2, 4, 6, 3, 5, 7, 1)
    x = x.reshape(B, l * h * w, t_size * p_size * p_size * C)
    return x

def unpatchify(x, p_size, t_size, H, W, T):
    B, L, patch_dim = x.shape
    C = patch_dim // (t_size * p_size * p_size)
    h, w, l = H // p_size, W // p_size, max(1, T // t_size)
    t_size = min(T, t_size)
    
    x = x.reshape(B, l, h, w, t_size, p_size, p_size, C)
    x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)
    x = x.reshape(B, C, T, H, W)
    return x

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fname", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint model")
    parser.add_argument("--output_dir", type=str, default="./eval/decoder_outputs")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of video distinct clips to process")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.fname, "r") as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    
    # ------------------
    # Model Setup
    # ------------------
    cfgs_model = params["model"]
    cfgs_data = params["data"]
    
    tubelet_size = cfgs_data.get("tubelet_size", 2)
    patch_size = cfgs_data.get("patch_size", 16)
    crop_size = cfgs_data.get("crop_size", 256)
    max_num_frames = max(cfgs_data.get("dataset_fpcs", [16]))
    model_name = cfgs_model.get("model_name", "vit_giant_xformers")
    
    encoder, predictor = init_video_model(
        device=device,
        patch_size=patch_size,
        max_num_frames=max_num_frames,
        tubelet_size=tubelet_size,
        model_name=model_name,
        crop_size=crop_size,
        pred_depth=cfgs_model.get("pred_depth", 6),
        pred_num_heads=cfgs_model.get("pred_num_heads", 12),
        pred_embed_dim=cfgs_model.get("pred_embed_dim", 384),
        is_causal=cfgs_model.get("is_causal", False),
        pred_is_causal=cfgs_model.get("pred_is_causal", False),
        use_sdpa=params.get("meta", {}).get("use_sdpa", False),
        has_cls_first=cfgs_model.get("has_cls_first", False),
        interpolate_rope=cfgs_model.get("interpolate_rope", False),
        img_temporal_dim_size=cfgs_model.get("img_temporal_dim_size", None),
        modality_embedding=cfgs_model.get("modality_embedding", False),
        n_registers=cfgs_model.get("n_registers", 0),
        n_registers_predictor=cfgs_model.get("n_registers_predictor", 0),
        use_rope=cfgs_model.get("use_rope", False),
        use_mask_tokens=cfgs_model.get("use_mask_tokens", False),
        num_mask_tokens=int(len(params.get("mask", [])) * len(cfgs_data.get("dataset_fpcs", [1]))),
        zero_init_mask_tokens=cfgs_model.get("zero_init_mask_tokens", True),
        use_silu=cfgs_model.get("use_silu", False),
        use_pred_silu=cfgs_model.get("use_pred_silu", False),
        wide_silu=cfgs_model.get("wide_silu", False),
        return_all_tokens=params.get("loss", {}).get("predict_all", True),
        chop_last_n_tokens=params.get("loss", {}).get("shift_by_n", 0),
        init_type=cfgs_model.get("init_type", "default")
    )
    
    levels_predictor = cfgs_model.get("n_output_distillation", 4)
    decoder_input_dim = encoder.backbone.embed_dim * levels_predictor
    patch_dim = 3 * tubelet_size * patch_size * patch_size

    decoder = PatchDiffusionDecoder(
        embed_dim=decoder_input_dim,
        patch_dim=patch_dim,
        timesteps=1000
    ).to(device)

    # ------------------
    # Load Checkpoint
    # ------------------
    print(f"Loading checkpoint {args.ckpt}...")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    
    def _strip_module(state_dict):
        return {k.replace("module.", ""): v for k, v in state_dict.items()}
        
    # Load strict=False just in case some other non-critical params were omitted
    encoder.load_state_dict(_strip_module(ckpt["encoder"]), strict=False)
    predictor.load_state_dict(_strip_module(ckpt["predictor"]), strict=False)
    
    if "decoder" in ckpt:
        decoder.load_state_dict(_strip_module(ckpt["decoder"]))
        print("Decoder weights successfully loaded!")
    else:
        print("WARNING: 'decoder' key not found in checkpoint. Generated images will be pure noise.")

    encoder.eval()
    predictor.eval()
    decoder.eval()

    # ------------------
    # Data Setup
    # ------------------
    transform = make_transforms(
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(1.0, 1.0),
        random_resize_scale=(1.0, 1.0),
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
        crop_size=crop_size,
    )
    
    mask_collator = MaskCollator(
        cfgs_mask=params["mask"],
        dataset_fpcs=cfgs_data["dataset_fpcs"],
        crop_size=crop_size,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
    )
    
    loader, _ = init_data(
        data=cfgs_data.get("dataset_type", "videodataset"),
        root_path=cfgs_data["datasets"],
        batch_size=args.num_samples,
        training=False,
        dataset_fpcs=cfgs_data["dataset_fpcs"],
        fps=cfgs_data.get("fps", 4),
        transform=transform,
        rank=0,
        world_size=1,
        datasets_weights=cfgs_data.get("datasets_weights", [1.0]),
        collator=mask_collator,
        num_workers=4,
        pin_mem=True,
    )
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1).to(device)
    def denormalize(x):
        return x * std + mean
        
    print(f"\nProcessing {args.num_samples} sample(s)...")
    sample = next(iter(loader))
    
    all_clips, all_masks_enc, all_masks_pred = [], [], []
    for fpc_sample in sample:
        udata, masks_enc_, masks_pred_ = fpc_sample
        all_clips += [udata[0][0].to(device)]
        all_masks_enc += [[m.to(device) for m in masks_enc_]]
        all_masks_pred += [[m.to(device) for m in masks_pred_]]
        
    clips = all_clips
    masks_enc = all_masks_enc
    masks_pred = all_masks_pred
    
    clip = clips[0]
    B, C, T, H, W = clip.shape
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            z = encoder(clips, masks_enc, gram_mode=False, training_mode=True)
            z_pred, _ = predictor(z, masks_enc, masks_pred)
            
            z_fpc = z_pred[0]
            masks_fpc = masks_pred[0]
            
        target_patches = patchify(clip, patch_size, tubelet_size)
        decoded_full_video_patches = target_patches.clone().float()
        
        for zij, mask_ij in zip(z_fpc, masks_fpc):
            B_z, K_z, D_z = zij.shape
            c_flat = zij.reshape(-1, D_z).float() # デコーダへはfloat32渡し
            
            print(f"-> Diffusion Sampling: {K_z} patches... please wait for reverse process!")
            # サンプリングはautocastブロックの外（float32のまま）で行う
            gen_patches_flat = decoder.sample(c_flat, shape=(B_z * K_z, patch_dim))
            
            print(f"DEBUG: gen_patches_flat std: {gen_patches_flat.std().item():.3f}, mean: {gen_patches_flat.mean().item():.3f}, min: {gen_patches_flat.min().item():.3f}, max: {gen_patches_flat.max().item():.3f}")
            
            gen_patches = gen_patches_flat.reshape(B_z, K_z, patch_dim)
            
            for b in range(B_z):
                decoded_full_video_patches[b, mask_ij[b]] = gen_patches[b]
                    
        decoded_clip = unpatchify(decoded_full_video_patches.float(), patch_size, tubelet_size, H, W, T)
        
    # Save visualizations
    clip_denorm = denormalize(clip.float()).clamp(0, 1)
    decoded_clip_denorm = denormalize(decoded_clip).clamp(0, 1)
    
    for b in range(B):
        for t in range(T):
            gt_frame = clip_denorm[b, :, t]
            pred_frame = decoded_clip_denorm[b, :, t]
            
            # Draw borders: Red border around predictions might be hard, so just side-by-side:
            # Horizontal cat: [Ground Truth (left) | Decoded Model Output (right)]
            combined = torch.cat([gt_frame, pred_frame], dim=-1)
            
            save_path = os.path.join(args.output_dir, f"sample_{b}_frame_{t:02d}.png")
            torchvision.utils.save_image(combined, save_path)
            
        print(f"✓ Saved visual comparisons for sequence {b} into '{args.output_dir}/sample_{b}_frame_*.png'")
        
    print("\nEvaluations Finished! Please check the output directory for side-by-side images.")

if __name__ == "__main__":
    main()

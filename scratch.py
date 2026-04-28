import sys
sys.path.append("/home/thoth-22/kei_ws/vjepa2")
from app.vjepa_2_1.utils import init_video_model
encoder, predictor, target_encoder = init_video_model(device="cpu", model_name="vit_giant_xformers")
print("embed_dim:", encoder.backbone.embed_dim)
print("predictor proj:", predictor.predictor_proj.out_features)

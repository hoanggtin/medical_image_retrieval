import numpy as np
from PIL import Image
from torchvision import transforms
import torch
import faiss
from config_cnn import Config
from cnn_fpn_rmac import CNN_FPN_RMAC
from faiss_helper import build_faiss_index, search_index
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Cấu hình
cfg = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = CNN_FPN_RMAC(cfg).to(device)
model.load_state_dict(torch.load(cfg.MODEL_SAVE_PATH, map_location=device))
model.eval()

# Load features
data = np.load("features/cnn_features.npz", allow_pickle=True)
features = data["features"]
paths = data["paths"]
labels = data["labels"] 

# Tạo index
index = build_faiss_index(features, features.shape[1])

# Trích xuất đặc trưng ảnh query
def extract_query_feature(img_path):
    image = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    x = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model.extract_features(x)
    return feat.squeeze(0).cpu().numpy()

# Query ảnh
query_path = r"D:\Job\test\lung\01cad8d0-45cd-4603-b099-94055d322310.png"  # ảnh cần truy vấn
query_feat = extract_query_feature(query_path)

D, I = search_index(index, query_feat.reshape(1, -1), k=6)
print("🔍 Top 5 ảnh gần nhất:")
for rank, idx in enumerate(I):
    print(f"{rank+1}. {paths[idx]} (distance={D[rank]:.4f})  {labels[idx]}")


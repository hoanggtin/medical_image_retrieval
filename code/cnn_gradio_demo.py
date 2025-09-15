import os
import torch
import numpy as np
from PIL import Image
import gradio as gr
from torchvision import transforms
from cnn_fpn_rmac import CNN_FPN_RMAC
from config_cnn import Config
from faiss_helper import build_faiss_index, search_index

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Cấu hình và model
cfg = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN_FPN_RMAC(cfg).to(device)
model.load_state_dict(torch.load(cfg.MODEL_SAVE_PATH, map_location=device))
model.eval()

# Load feature database
data = np.load("features/cnn_scratch_features.npz", allow_pickle=True)
features = data["features"]
paths = data["paths"]
labels = data["labels"] 

# Tạo FAISS index cho toàn bộ dataset
global_index = build_faiss_index(features, features.shape[1])

# Transform ảnh
transform = transforms.Compose([
    transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

# Hàm trích xuất đặc trưng từ ảnh
def extract_feature(img):
    img = img.convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model.extract_features(x)
    return feat.squeeze(0).cpu().numpy().reshape(1, -1)

# Hàm phân loại ảnh (chỉ để hiển thị thông tin)
def classify_image(img):
    img = img.convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs).item()
        confidence = probs[0][pred_class].item()
    return pred_class, confidence

# Hàm truy vấn ảnh tương tự
def retrieve_similar_images(query_img, top_k):
    # Bước 1: Phân loại ảnh (để hiển thị)
    pred_class, confidence = classify_image(query_img)
    class_names = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]

    # Bước 2: Trích xuất đặc trưng và truy vấn toàn bộ ảnh
    query_feat = extract_feature(query_img)
    D, I = search_index(global_index, query_feat, k=top_k)

    results = []
    for rank, idx in enumerate(I):
        img_path = paths[idx]
        img_class = labels[idx]
        class_name = class_names[img_class]
        distance = D[rank]

        img = Image.open(img_path).resize((224, 224))
        caption = f"Class: {class_name} | Distance: {distance:.4f}"
        results.append((img, caption))
    top1_class = class_names[labels[I[0]]]
    top1_dist = D[0]
    top1_text = f"Predicted class: {top1_class} (Distance: {top1_dist:.4f})"

    return results, f"Predicted class: {top1_class} (Confidence: {top1_dist:.4f})"

# Giao diện Gradio
demo = gr.Interface(
    fn=retrieve_similar_images,
    inputs=[
        gr.Image(type="pil", label="Upload X-ray image"),
        gr.Slider(minimum=1, maximum=20, value=6, step=1, label="Top K similar images")
    ],
    outputs=[
        gr.Gallery(label="Top similar images"),
        gr.Text(label="Classification Result")
    ],
    title="Medical Image Retrieval with CNN + FAISS",
    description="Upload a chest X-ray image and select how many similar images you want to retrieve."
)

if __name__ == "__main__":
    demo.launch()

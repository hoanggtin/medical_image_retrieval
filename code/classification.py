import torch
from torchvision import transforms
from PIL import Image
from config import Config
from vit_fpn_rmac import ViT_FPN_RMAC

# Load model
cfg = Config()
cfg.NUM_CLASSES = 4
model = ViT_FPN_RMAC(cfg)

# Sửa dòng này - thêm map_location
model_path = "D:\\Job\\image_retrieval\\models\\covid_vit_fpn_rmac.pth"
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))  # Hoặc 'cuda' nếu có GPU

model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define class names
class_names = ["COVID", "Normal", "Viral Pneumonia", "Lung_Opacity"]

# Define transforms (same as during training)
def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def classify_image(image_path):
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    transform = get_transforms()
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        pred_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][pred_class].item()
    
    return class_names[pred_class], confidence

# Example usage
if __name__ == "__main__":
    image_path = r"D:\Job\image_retrieval\data\COVID_Dataset\Viral Pneumonia\Viral Pneumonia-178.png"  # Thay bằng đường dẫn ảnh thực tế
    predicted_class, confidence = classify_image(image_path)
    print(f"Predicted class: {predicted_class} with confidence: {confidence:.4f}")
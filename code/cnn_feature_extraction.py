import os
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from config_cnn import Config
from dataset import MedicalDataset
from cnn_fpn_rmac import CNN_FPN_RMAC  

def extract_features_for_dataset(model, dataloader, device):
    model.eval()
    all_features = []
    all_labels = []
    image_paths = []

    with torch.no_grad():
        for imgs, labels, paths in tqdm(dataloader, desc="🔍 Extracting Features"):
            imgs = imgs.to(device)
            feats = model.extract_features(imgs)  # ⚠️ dùng .extract_features() thay vì forward
            all_features.append(feats.cpu().numpy())
            all_labels.extend(labels.numpy())
            image_paths.extend(paths)

    features = np.concatenate(all_features, axis=0)
    return features, np.array(all_labels), image_paths

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config()

    # Load CNN model
    model = CNN_FPN_RMAC(cfg).to(device)
    model.load_state_dict(torch.load(r"D:\Job\image_retrieval\models\checkpoints\best_cnn_model.pth", map_location=device))
    model.eval()

    # Image transform
    transform = transforms.Compose([
        transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

    # Dataset có path trả về
    class IndexedMedicalDataset(MedicalDataset):
        def __getitem__(self, idx):
            img_path = self.image_paths[idx]
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            label = self.labels[idx]
            return image, label, img_path

    dataset = IndexedMedicalDataset(cfg.DATA_DIR, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=False)

    # Extract features
    feats, labels, paths = extract_features_for_dataset(model, dataloader, device)

    # Save
    os.makedirs("features", exist_ok=True)
    np.savez("features/cnn_scratch_features.npz", features=feats, labels=labels, paths=paths)
    print("✅ Feature extraction done. Saved to features/cnn_scratch_features.npz")

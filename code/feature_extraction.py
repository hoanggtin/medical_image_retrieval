import os
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from vit_fpn_rmac import ViT_FPN_RMAC
from config import Config
from dataset import MedicalDataset

def extract_features_for_dataset(model, dataloader, device):
    model.eval()
    all_features = []
    all_labels = []
    image_paths = []

    with torch.no_grad():
        for imgs, labels, paths in tqdm(dataloader, desc="🔍 Extracting Features"):
            imgs = imgs.to(device)
            feats = model(imgs)  # shape: (B, feat_dim)
            all_features.append(feats.cpu().numpy())
            all_labels.extend(labels.numpy())
            image_paths.extend(paths)

    features = np.concatenate(all_features, axis=0)
    return features, np.array(all_labels), image_paths

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config()

    # Load model
    model = ViT_FPN_RMAC(cfg).to(device)
    model.load_state_dict(torch.load(cfg.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    # Prepare dataset
    transform = transforms.Compose([
        transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

    # Custom dataset để lấy cả image paths
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

    # Extract
    feats, labels, paths = extract_features_for_dataset(model, dataloader, device)

    # Save
    os.makedirs("features", exist_ok=True)
    np.savez("features/vit_features.npz", features=feats, labels=labels, paths=paths)
    print("✅ Feature extraction done. Saved to features/vit_features.npz")

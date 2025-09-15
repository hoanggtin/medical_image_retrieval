
from PIL import Image
import os
from torch.utils.data import Dataset
from torchvision import transforms

def get_transforms(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

class MedicalDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.classes = sorted(os.listdir(image_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        valid_exts = [".jpg", ".jpeg", ".png"]

        for cls in self.classes:
            cls_folder = os.path.join(image_dir, cls)
            for fname in os.listdir(cls_folder):
                fpath = os.path.join(cls_folder, fname)
                if os.path.isfile(fpath) and os.path.splitext(fname)[1].lower() in valid_exts:
                    self.image_paths.append(fpath)
                    self.labels.append(self.class_to_idx[cls])


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

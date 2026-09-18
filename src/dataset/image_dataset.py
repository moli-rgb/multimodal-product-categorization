import torch 
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms
import duckdb
from src.dataset.text_dataset import load_train_data

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class ImageDataset(Dataset):
    def __init__(self,image_path,labels,category2idx,transform):
        self.image_path = image_path
        self.labels = labels
        self.category2idx = category2idx
        self.transform = transform
    def __len__(self):
        return len(self.image_path)
    def __getitem__(self, idx):
        image = Image.open(self.image_path[idx]).convert('RGB')
        label = self.category2idx[self.labels[idx]]
        if self.transform:
            image = self.transform(image)
        return image, label

def load_image_data(split, category2idx, batch_size=4):
    conn = duckdb.connect(database='catalog.db', read_only=False)
    cursor = conn.cursor()
    data = cursor.execute("""
        SELECT 
            product_images.image_path,
            products.category_id 
        FROM product_images
        JOIN products ON products.product_id = product_images.product_id 
        WHERE products.split = ? 
          AND product_images.image_path IS NOT NULL 
          AND products.category_id IS NOT NULL
          
    """, (split,)).fetchall()
    
    image_paths, labels = zip(*data)
    
    dataset = ImageDataset(list(image_paths), list(labels), category2idx, image_transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train'))
    
    return loader

if __name__ == "__main__":
    _, vocab, category2idx = load_train_data()
    
    train_loader = load_image_data('train', category2idx)
    batch_images, batch_labels = next(iter(train_loader))
    print("Batch images shape:", batch_images.shape)
    print("Batch labels shape:", batch_labels.shape)
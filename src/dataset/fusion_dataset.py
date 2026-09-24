import torch
from torch.utils.data import Dataset
from PIL import Image
import duckdb
from torchvision import transforms
from src.preprocessing.vocab import Vocabulary
from src.preprocessing.text_cleaner import clean_text

class FusionDataset(Dataset):
    def __init__(self, titles, image_paths, labels, vocab, category2idx, transform):
        self.titles = titles
        self.image_paths = image_paths
        self.labels = labels
        self.vocab = vocab
        self.category2idx = category2idx
        self.transform = transform

    def __len__(self):
        return len(self.titles)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        text = self.titles[idx]
        label = self.labels[idx]

        text_ids = self.vocab.encode(text)
        if self.transform:
            image = self.transform(image)
        label_idx = self.category2idx[label]

        return (
            torch.tensor(text_ids, dtype=torch.long),
            image,
            torch.tensor(label_idx, dtype=torch.long)
        )
    
def load_fusion_data(split):
    conn = duckdb.connect(database='catalog.db', read_only=True)
    data = conn.execute("""
        SELECT 
            products.title,
            product_images.image_path,
            products.category_id
        FROM products
        JOIN product_images ON products.product_id = product_images.product_id
        WHERE products.split = ?
          AND products.title IS NOT NULL AND products.title != ''
          AND products.category_id IS NOT NULL
          AND product_images.image_path IS NOT NULL
          AND product_images.image_order = 1
        QUALIFY ROW_NUMBER() OVER (PARTITION BY products.product_id ORDER BY products.product_id) = 1
    """, (split,)).fetchall()

    titles, image_paths, labels = zip(*data)
    cleaned_title=list(map(clean_text, titles))
    return list(cleaned_title), list(image_paths), list(labels)

def fusion_collate_fn(batch):
    texts, images, labels = zip(*batch)
    padded_texts = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=0)
    images = torch.stack(images)
    labels = torch.stack(labels)

    return padded_texts, images, labels

if __name__ == "__main__":
    cleaned_title, image_paths, labels = load_fusion_data('train')     
    vocab = Vocabulary(cleaned_title, min_freq=2) 
    unique_categories = sorted(set(labels))
    category2idx = {cat: idx for idx, cat in enumerate(unique_categories)}
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = FusionDataset(cleaned_title, image_paths, labels, vocab, category2idx, image_transform)
    
    text_ids, image, label_idx = dataset[0]
    print(f"Shape of text_ids: {text_ids.shape}")
    print(f"Image shape: {image.shape}")
    print(f"Label index: {label_idx}")
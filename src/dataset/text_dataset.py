import torch
from torch.utils.data import Dataset,DataLoader
import duckdb
from src.preprocessing.text_cleaner import clean_text
from src.preprocessing.vocab import Vocabulary

class TextCatalogDataset(Dataset):
    def __init__(self, texts, labels, vocab, category2idx):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.category2idx = category2idx

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        text_ids = self.vocab.encode(text)
        label_idx = self.category2idx[label]
        return torch.tensor(text_ids, dtype=torch.long), torch.tensor(label_idx, dtype=torch.long)

def collate_fn(batch):
    texts, labels = zip(*batch)
    lengths = torch.tensor([len(text) for text in texts], dtype=torch.long)
    padded_texts = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    return padded_texts, lengths, labels

def load_train_data():
    conn = duckdb.connect(database='catalog.db', read_only=False)
    cursor = conn.cursor()
    train_data = conn.execute("""
        SELECT title, category_id 
        FROM products 
        WHERE split = 'train' AND title IS NOT NULL AND title != '' AND category_id IS NOT NULL
    """).fetchall()
    train_text, train_labels = zip(*train_data)
    cleaned_train_text = list(map(clean_text, train_text))

    vocab = Vocabulary(cleaned_train_text, min_freq=2)

    unique_categories = sorted(set(train_labels))
    category2idx = {cat: idx for idx, cat in enumerate(unique_categories)}

    dataset = TextCatalogDataset(cleaned_train_text, train_labels, vocab, category2idx)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    return loader, vocab, category2idx

def load_val_data(vocab, category2idx):
    conn = duckdb.connect(database='catalog.db', read_only=False)
    cursor = conn.cursor()
    val_data = conn.execute("""
        SELECT title, category_id 
        FROM products 
        WHERE split = 'val' AND title IS NOT NULL AND title != '' AND category_id IS NOT NULL
    """).fetchall()
    val_text, val_labels = zip(*val_data)
    cleaned_val_text = list(map(clean_text, val_text))

    dataset = TextCatalogDataset(cleaned_val_text, val_labels, vocab, category2idx)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    return loader
if __name__ == "__main__":
    loader, vocab, category2idx = load_train_data()
    batch_texts, batch_lengths, batch_labels = next(iter(loader))
    print("Batch texts shape:", batch_texts.shape)
    

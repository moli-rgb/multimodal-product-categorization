import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torchvision import models
from src.dataset.fusion_dataset import FusionDataset, load_fusion_data, fusion_collate_fn
from src.preprocessing.vocab import Vocabulary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(input_size=embedding_dim,
                          hidden_size=hidden_dim, batch_first=True)

    def forward(self, text_ids):
        embedded = self.embedding(text_ids)
        _, hidden = self.gru(embedded)
        return hidden.squeeze(0)


class ImageEncoder(nn.Module):
    def __init__(self, output_dim=512):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.proj = nn.Linear(in_features, output_dim)
        for name, param in self.backbone.named_parameters():
            param.requires_grad = 'layer4' in name
            
    def forward(self, images):
        features = self.backbone(images)
        return self.proj(features)

def get_optimizer(model, backbone_lr=1e-5, head_lr=1e-3):
    backbone_params = model.image_encoder.backbone.parameters()
    other_params = (
        list(model.image_encoder.proj.parameters()) +
        list(model.text_encoder.parameters()) +
        list(model.classifier.parameters())
    )
    return torch.optim.Adam([
        {"params": backbone_params, "lr": backbone_lr},
        {"params": other_params, "lr": head_lr},
    ])

class FusionClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, image_feature_dim, num_classes, dropout=0.2):
        super().__init__()
        self.text_encoder = TextEncoder(vocab_size, embedding_dim, hidden_dim)
        self.image_encoder = ImageEncoder(output_dim=image_feature_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + image_feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, text_ids, images):
        text_features = self.text_encoder(text_ids)
        image_features = self.image_encoder(images)
        fused = torch.cat([text_features, image_features], dim=1)
        return self.classifier(fused)


def evaluate_fusion_model(model, val_loader):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for texts, images, labels in val_loader:
            texts = texts.to(device)
            images = images.to(device)
            labels = labels.to(device)

            output = model(texts, images)
            predictions = output.argmax(dim=1)
            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    return f1_score(all_labels, all_predictions, average='macro', zero_division=0)


def train_fusion_model(model, train_loader, val_loader, num_epochs=10, learning_rate=1e-3, patience=3):
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, backbone_lr=1e-5, head_lr=learning_rate)

    best_f1 = 0.0
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for texts, images, labels in train_loader:
            texts = texts.to(device)
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(texts, images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(len(train_loader), 1)
        val_f1 = evaluate_fusion_model(model, val_loader)
        print(
            f"Epoch {epoch + 1:02d} | loss={avg_loss:.4f} | val_macro_f1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), 'best_fusion_model.pth')
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break
    return best_f1


if __name__ == "__main__":
    train_titles, train_image_paths, train_labels = load_fusion_data('train')
    val_titles, val_image_paths, val_labels = load_fusion_data('val')

    vocab = Vocabulary(train_titles, min_freq=2)    
    categories = sorted(set(train_labels))
    category2idx = {category: idx for idx, category in enumerate(categories)}

    image_transform = models.ResNet18_Weights.DEFAULT.transforms()
    train_dataset = FusionDataset(
        train_titles, train_image_paths, train_labels, vocab, category2idx, image_transform)
    val_dataset = FusionDataset(
        val_titles, val_image_paths, val_labels, vocab, category2idx, image_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=16, shuffle=True, collate_fn=fusion_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16,
                            shuffle=False, collate_fn=fusion_collate_fn)

    model = FusionClassifier(
        vocab_size=len(vocab),
        embedding_dim=64,
        hidden_dim=256,
        image_feature_dim=256,
        num_classes=len(category2idx),
    ).to(device)

    train_fusion_model(model, train_loader, val_loader,
                       num_epochs=1, learning_rate=1e-3, patience=3)
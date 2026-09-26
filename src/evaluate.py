import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torchvision import models

from src.dataset.text_dataset import load_train_data, load_val_data
from src.dataset.image_dataset import load_image_data
from src.dataset.fusion_dataset import FusionDataset, load_fusion_data, fusion_collate_fn
from src.models.text_model import TextClassifier
from src.models.image_model import ImageClassifier
from src.models.fusion_model import FusionClassifier
from src.preprocessing.vocab import Vocabulary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate_text_model():
    _, vocab, category2idx = load_train_data()
    test_loader = load_val_data('test', vocab, category2idx)

    model = TextClassifier(vocab_size=len(
        vocab), embedding_dim=50, hidden_dim=64, num_classes=len(category2idx))
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    model.to(device)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for texts, lengths, labels in test_loader:
            texts, labels = texts.to(device), labels.to(device)
            output = model(texts)
            preds = output.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    print(f"GRU macro-F1: {f1:.4f}")
    return f1


def evaluate_image_model():
    _, vocab, category2idx = load_train_data()
    test_loader = load_image_data('test', category2idx, batch_size=32)

    model = ImageClassifier(num_classes=len(category2idx))
    model.load_state_dict(torch.load(
        'best_image_model.pth', map_location=device))
    model.to(device)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            preds = output.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    print(f"Image macro-F1: {f1:.4f}")
    return f1


def evaluate_fusion_model():
    train_titles, _, train_labels = load_fusion_data('train')
    test_titles, test_image_paths, test_labels = load_fusion_data('test')

    vocab = Vocabulary(train_titles, min_freq=2)
    categories = sorted(set(train_labels))
    category2idx = {c: i for i, c in enumerate(categories)}

    image_transform = models.ResNet18_Weights.DEFAULT.transforms()
    test_dataset = FusionDataset(
        test_titles, test_image_paths, test_labels, vocab, category2idx, image_transform)
    test_loader = DataLoader(test_dataset, batch_size=16,
                             shuffle=False, collate_fn=fusion_collate_fn)

    model = FusionClassifier(
        vocab_size=len(vocab),
        embedding_dim=64,
        hidden_dim=256,
        image_feature_dim=256,
        num_classes=len(category2idx),
    )
    model.load_state_dict(torch.load(
        'best_fusion_model.pth', map_location=device))
    model.to(device)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for texts, images, labels in test_loader:
            texts, images, labels = texts.to(
                device), images.to(device), labels.to(device)
            output = model(texts, images)
            preds = output.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    print(f"Fusion macro-F1: {f1:.4f}")
    return f1


if __name__ == "__main__":
    text_f1 = evaluate_text_model()
    image_f1 = evaluate_image_model()
    fusion_f1 = evaluate_fusion_model()

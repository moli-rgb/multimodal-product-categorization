import torch
import torch.nn as nn
from torchvision import models
from src.dataset.text_dataset import load_train_data
from src.dataset.image_dataset import load_image_data
from sklearn.metrics import f1_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ImageClassifier(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
    def forward(self, x):
        return self.backbone(x)

def evaluate_image_model(model, val_loader):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch_images, batch_labels in val_loader:
            batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
            output = model(batch_images)
            predicted_classes = output.argmax(dim=1)
            all_predictions.extend(predicted_classes.tolist())
            all_labels.extend(batch_labels.tolist())

    macro_f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
    return macro_f1

if __name__ == "__main__":
    _, vocab, category2idx = load_train_data()
    train_loader = load_image_data('train', category2idx, batch_size=32)
    val_loader = load_image_data('val', category2idx, batch_size=32)

    model = ImageClassifier(num_classes=len(category2idx)).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.backbone.fc.parameters(), lr=0.001)

    num_epochs = 10
    best_f1 = 0
    epochs_without_improvement = 0
    patience = 3
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_images, batch_labels in train_loader:
            batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            output = model(batch_images)
            loss = criterion(output, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss}")

        val_f1 = evaluate_image_model(model, val_loader)
        print(f"  → Validation macro-F1: {val_f1:.4f}")
        model.train()

        if val_f1 > best_f1:
            best_f1 = val_f1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), 'best_image_model.pth')
            print(f"  → New best model saved! F1: {best_f1:.4f}")
        else:
            epochs_without_improvement += 1
            print(f"  → No improvement for {epochs_without_improvement} epoch(s)")

        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

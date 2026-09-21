import torch
import torch.nn as nn
from src.dataset.text_dataset import load_train_data,load_val_data
from sklearn.metrics import f1_score   

class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        _, hidden = self.gru(embedded)
        output = self.fc(hidden.squeeze(0))
        return output

def evaluate(model, val_loader):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch_texts, batch_lengths, batch_labels in val_loader:
            output = model(batch_texts)
            predicted_classes = output.argmax(dim=1)
            all_predictions.extend(predicted_classes.tolist())
            all_labels.extend(batch_labels.tolist())

    macro_f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
    return macro_f1

if __name__ == "__main__":
    loader, vocab, category2idx = load_train_data()
    val_loader = load_val_data(vocab, category2idx)     
    model = TextClassifier(vocab_size=len(vocab), embedding_dim=50, hidden_dim=64, num_classes=len(category2idx))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 30

    best_f1 = 0
    epochs_without_improvement = 0
    patience = 3

    for epoch in range(num_epochs):
        total_loss = 0
        for batch_texts, batch_lengths, batch_labels in loader:
            optimizer.zero_grad()
            output = model(batch_texts)
            loss = criterion(output, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss}")

        val_f1 = evaluate(model, val_loader)
        print(f"  → Validation macro-F1: {val_f1:.4f}")
        model.train()

        if val_f1 > best_f1:
            best_f1 = val_f1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"  → New best model saved! F1: {best_f1:.4f}")
        else:
            epochs_without_improvement += 1
            print(f"  → No improvement for {epochs_without_improvement} epoch(s)")

        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report

def train_model(mlp_model, train_loader, criterion, optimizer, device, num_epochs=5):
    for epoch in range(num_epochs):
        mlp_model.train()
        train_loss = 0.0

        for batch in train_loader:
            embeddings = batch['embedding'].to(device)
            labels = batch['labels'].to(device)
            embeddings = torch.mean(embeddings, dim=1)
            
            optimizer.zero_grad()
            outputs = mlp_model(embeddings)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_train_loss}")

import torch



def evaluate_model(model, criterion, dataloader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            embeddings = batch['embedding'].to(device)
            labels = batch['labels'].to(device)
            embeddings = torch.mean(embeddings, dim=1)  # Ensure shape consistency

            outputs = model(embeddings)  # Forward pass

            loss = criterion(outputs, labels)  # Compute loss

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    return avg_loss, accuracy       
# _______________________________________________________________________-
# import torch
# import torch.optim as optim
# import torch.nn as nn
# from sklearn.metrics import accuracy_score, classification_report
# from collections import Counter
# from torch.utils.data import DataLoader, WeightedRandomSampler

# def get_weighted_sampler(train_labels):
#     """ Compute class weights and create a WeightedRandomSampler to handle class imbalance. """
#     class_counts = Counter(train_labels)
#     total_samples = sum(class_counts.values())

#     class_weights = {label: total_samples / count for label, count in class_counts.items()}
#     sample_weights = [class_weights[label] for label in train_labels]

#     return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

# def train_model(mlp_model, train_loader, criterion, optimizer, device, num_epochs=5):
#     """ Trains the MLP model with weighted sampling to balance classes. """
#     for epoch in range(num_epochs):
#         mlp_model.train()
#         train_loss = 0.0

#         for batch in train_loader:
#             embeddings = batch['embedding'].to(device)
#             labels = batch['labels'].to(device)
#             embeddings = embeddings[:, 0, :]  # Use CLS token for consistency

#             optimizer.zero_grad()
#             outputs = mlp_model(embeddings)

#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.item()

#         avg_train_loss = train_loss / len(train_loader)
#         print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_train_loss:.4f}")

#     # ✅ Save the trained model
#     torch.save(mlp_model.state_dict(), "mlp_model.pth")
#     print("✅ Model saved successfully.")

# def evaluate_model(model, criterion, dataloader, device):
#     """ Evaluates the model and applies softmax for proper probability distribution. """
#     model.eval()
#     total_loss = 0.0
#     correct = 0
#     total = 0

#     with torch.no_grad():
#         for batch in dataloader:
#             embeddings = batch['embedding'].to(device)
#             labels = batch['labels'].to(device)
#             embeddings = embeddings[:, 0, :]  # Use CLS token for consistency

#             outputs = model(embeddings)
#             probabilities = torch.softmax(outputs, dim=1)  # Apply softmax
#             preds = torch.argmax(probabilities, dim=1)

#             loss = criterion(outputs, labels)
#             total_loss += loss.item()
#             correct += (preds == labels).sum().item()
#             total += labels.size(0)

#     avg_loss = total_loss / len(dataloader)
#     accuracy = correct / total

#     print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy * 100:.2f}%")
#     return avg_loss, accuracy

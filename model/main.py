import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoModel
from data_processing import load_data, EmotionDataset
from model import MLPClassifier
from training_model import train_model, evaluate_model

def main():
    file_path = 'data/outputfinal.csv'
    model_name = "ai4bharat/indic-bert"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_texts, val_texts, test_texts, train_labels, val_labels, test_labels, label_encoder, tokenizer = load_data(file_path, model_name, device)

    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    model.to(device)
    model.eval()

    # ✅ Pass label_encoder while creating EmotionDataset
    train_dataset = EmotionDataset(train_texts, train_labels, tokenizer, model, device, label_encoder)
    val_dataset = EmotionDataset(val_texts, val_labels, tokenizer, model, device, label_encoder)
    test_dataset = EmotionDataset(test_texts, test_labels, tokenizer, model, device, label_encoder)

    batch_size = 10
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    input_dim = model.config.hidden_size
    hidden_dim1 = 256
    hidden_dim2 = 128
    output_dim = len(label_encoder.classes_)
    dropout_prob = 0.3

    mlp_model = MLPClassifier(input_dim, hidden_dim1, hidden_dim2, output_dim, dropout_prob).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(mlp_model.parameters(), lr=0.001)

    num_epochs = 20
    train_model(mlp_model, train_loader, criterion, optimizer, device, num_epochs)

    val_loss, val_accuracy = evaluate_model(mlp_model, criterion, val_loader, device)
    print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}%")

    test_loss, test_accuracy = evaluate_model(mlp_model, criterion, test_loader, device)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}%")

    train_loss, train_accuracy = evaluate_model(mlp_model, criterion, train_loader, device)
    print(f"Train Loss: {train_loss}, Train Accuracy: {train_accuracy}%")

if __name__ == "__main__":
    main()
# ______________________________________________________
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from transformers import AutoModel
# from data_processing import load_data, EmotionDataset
# from model import MLPClassifier
# from training_model import train_model, evaluate_model, get_weighted_sampler

# def main():
#     file_path = 'data/outputfinal.csv'
#     model_name = "ai4bharat/indic-bert"
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     # Load Data
#     train_texts, val_texts, test_texts, train_labels, val_labels, test_labels, label_encoder, tokenizer = load_data(file_path, model_name, device)

#     # Load IndicBERT Model
#     model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
#     model.to(device)
#     model.eval()

#     # Create Emotion Dataset with label encoder
#     train_dataset = EmotionDataset(train_texts, train_labels, tokenizer, model, device, label_encoder)
#     val_dataset = EmotionDataset(val_texts, val_labels, tokenizer, model, device, label_encoder)
#     test_dataset = EmotionDataset(test_texts, test_labels, tokenizer, model, device, label_encoder)

#     batch_size = 10

#     #  Extract Labels Correctly for Weighted Sampler
#     train_labels_list = [train_dataset[i]['labels'].item() for i in range(len(train_dataset))]

#     sampler = get_weighted_sampler(train_labels_list)

#     # ✅ Create Data Loaders with Weighted Sampling
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#     # Define Model Architecture
#     input_dim = model.config.hidden_size
#     hidden_dim1 = 256
#     hidden_dim2 = 128
#     output_dim = len(label_encoder.classes_)
#     dropout_prob = 0.3

#     mlp_model = MLPClassifier(input_dim, hidden_dim1, hidden_dim2, output_dim, dropout_prob).to(device)
    
#     # Define Loss & Optimizer
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(mlp_model.parameters(), lr=0.001)

#     # ✅ Train Model
#     num_epochs = 10
#     train_model(mlp_model, train_loader, criterion, optimizer, device, num_epochs)

#     # ✅ Evaluate Model
#     val_loss, val_accuracy = evaluate_model(mlp_model, criterion, val_loader, device)
#     print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy * 100:.2f}%")

#     test_loss, test_accuracy = evaluate_model(mlp_model, criterion, test_loader, device)
#     print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy * 100:.2f}%")

#     train_loss, train_accuracy = evaluate_model(mlp_model, criterion, train_loader, device)
#     print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy * 100:.2f}%")

# if __name__ == "__main__":
#     main()

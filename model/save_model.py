import torch
from model import MLPClassifier  # Import the MLP model class

# Define model architecture (Ensure values match your training setup)
input_dim = 768
hidden_dim1 = 256  # Example hidden layer 1 size
hidden_dim2 = 128  # Example hidden layer 2 size
output_dim = 4  # Number of emotion classes
dropout_prob = 0.3  # Dropout probability

# Initialize the model
mlp_model = MLPClassifier(input_dim, hidden_dim1, hidden_dim2, output_dim, dropout_prob)

# Define model path
import os

model_dir = os.path.join(os.getcwd(), "model")  # Ensures correct path
os.makedirs(model_dir, exist_ok=True)  # Create model directory if it doesn't exist

model_path = os.path.join(model_dir, "mlp_model.pth")

# Save the model state_dict (only weights)
torch.save(mlp_model.state_dict(), model_path)

print(f"✅ Model successfully saved at: {model_path}")

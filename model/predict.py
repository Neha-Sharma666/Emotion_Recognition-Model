# import sys
# import os
# import torch
# from transformers import AutoModel, AutoTokenizer

# # ✅ Ensure the 'model' folder is correctly recognized
# sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# # Import MLPClassifier correctly
# from model.model import MLPClassifier  # Corrected path

#   # Import MLP model from model.py

# # Define Model and Tokenizer
# MODEL_NAME = "ai4bharat/indic-bert"
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load IndicBERT Model
# indicbert_model = AutoModel.from_pretrained(MODEL_NAME, output_hidden_states=True).to(device)
# indicbert_model.eval()

# # Load Trained MLP Model
# input_dim = indicbert_model.config.hidden_size
# hidden_dim1 = 256
# hidden_dim2 = 128
# output_dim = 4
# dropout_prob = 0.3

# # ✅ Ensure correct model path
# model_path = os.path.join(os.path.dirname(__file__), "mlp_model.pth")
# if not os.path.exists(model_path):
#     raise FileNotFoundError(f"🚨 Model file not found at {model_path}. Please run `save_model.py` first!")

# # Initialize and Load MLP Model
# mlp_model = MLPClassifier(input_dim, hidden_dim1, hidden_dim2, output_dim, dropout_prob).to(device)
# mlp_model.load_state_dict(torch.load(model_path, map_location=device))
# mlp_model.eval()

# print("✅ Model loaded successfully for prediction.")

# # Define Emotion Labels
# emotion_labels = {0: "Angry", 1: "Sad", 2: "Neutral", 3: "Happy"}

# # Function to Predict Emotion
# def predict_emotion(text):
#     encoding = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
#     input_ids = encoding["input_ids"].to(device)
#     attention_mask = encoding["attention_mask"].to(device)

#     with torch.no_grad():
#         outputs = indicbert_model(input_ids=input_ids, attention_mask=attention_mask)
#         hidden_states = outputs.hidden_states
#         second_to_last_layer = hidden_states[-2]
#         sentence_embedding = second_to_last_layer[:, 0, :]

#         emotion_logits = mlp_model(sentence_embedding)
#         predicted_label = torch.argmax(emotion_logits, dim=1).item()

#     return emotion_labels[predicted_label]

# # Test Example
# if __name__ == "__main__":
#     user_input = input("Enter a sentence: ")
#     emotion = predict_emotion(user_input)
#     print(f"Predicted Emotion: {emotion}")
# _______________________________________________________________
# import sys
# import os
# import torch
# from transformers import AutoModel, AutoTokenizer
# import torch.nn.functional as F 

# # ✅ Ensure 'model' folder is recognized
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# # ✅ Import MLPClassifier correctly
# from model.model import MLPClassifier  # Import from model.py

# # ✅ Define Model and Tokenizer
# MODEL_NAME = "ai4bharat/indic-bert"
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ✅ Load IndicBERT Model
# indicbert_model = AutoModel.from_pretrained(MODEL_NAME, output_hidden_states=True).to(device)
# indicbert_model.eval()

# # ✅ Load Trained MLP Model
# input_dim = indicbert_model.config.hidden_size  # 768 for IndicBERT
# hidden_dim1 = 256
# hidden_dim2 = 128
# output_dim = 4
# dropout_prob = 0.3

# # ✅ Ensure correct model path
# model_path = os.path.join(os.path.dirname(__file__), "mlp_model.pth")
# if not os.path.exists(model_path):
#     raise FileNotFoundError(f"🚨 Model file not found at {model_path}. Please run `save_model.py` first!")

# # ✅ Initialize and Load MLP Model
# mlp_model = MLPClassifier(input_dim, hidden_dim1, hidden_dim2, output_dim, dropout_prob).to(device)
# mlp_model.load_state_dict(torch.load(model_path, map_location=device))
# mlp_model.eval()

# print("✅ Model loaded successfully for prediction.")

# # ✅ Define Emotion Labels
# emotion_labels = {0: "Angry", 1: "Sad", 2: "Neutral", 3: "Happy"}

# # ✅ Function to Predict Emotion
# # def predict_emotion(text):
# #     encoding = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
# #     input_ids = encoding["input_ids"].to(device)
# #     attention_mask = encoding["attention_mask"].to(device)

# #     with torch.no_grad():
# #         outputs = indicbert_model(input_ids=input_ids, attention_mask=attention_mask)
# #         hidden_states = outputs.hidden_states
# #         second_to_last_layer = hidden_states[-2]
# #         sentence_embedding = second_to_last_layer[:, 0, :]

# #         emotion_logits = mlp_model(sentence_embedding)
# #         predicted_label = torch.argmax(emotion_logits, dim=1).item()

# #     return emotion_labels[predicted_label]
# def predict_emotion(text):
#     encoding = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
#     input_ids = encoding["input_ids"].to(device)
#     attention_mask = encoding["attention_mask"].to(device)

#     with torch.no_grad():
#         outputs = indicbert_model(input_ids=input_ids, attention_mask=attention_mask)
#         hidden_states = outputs.hidden_states
#         second_to_last_layer = hidden_states[-2]
#         sentence_embedding = second_to_last_layer[:, 0, :]

#         emotion_logits = mlp_model(sentence_embedding)
#         probabilities = F.softmax(emotion_logits, dim=1)  # Apply softmax
#         predicted_label = torch.argmax(probabilities, dim=1).item()

#         print("Predicted Probabilities:", probabilities.cpu().numpy())  # Debugging
#         print("Predicted Label:", predicted_label)

#     return emotion_labels[predicted_label]

# # ✅ Test Example (Optional)
# if __name__ == "__main__":
#     user_input = input("Enter a sentence: ")
#     emotion = predict_emotion(user_input)
#     print(f"Predicted Emotion: {emotion}")








# import sys
# import os
# import torch
# from transformers import AutoModel, AutoTokenizer
# import torch.nn.functional as F 

# # ✅ Ensure 'model' folder is recognized
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# # ✅ Import MLPClassifier correctly
# from model import MLPClassifier  # Import from model.py

# # ✅ Define Model and Tokenizer
# MODEL_NAME = "ai4bharat/indic-bert"
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ✅ Load IndicBERT Model
# indicbert_model = AutoModel.from_pretrained(MODEL_NAME, output_hidden_states=True).to(device)
# indicbert_model.eval()

# # ✅ Load Trained MLP Model
# input_dim = indicbert_model.config.hidden_size  # 768 for IndicBERT
# hidden_dim1 = 256
# hidden_dim2 = 128
# output_dim = 4
# dropout_prob = 0.3

# # ✅ Ensure correct model path
# model_path = "mlp_model.pth"
# if not os.path.exists(model_path):
#     raise FileNotFoundError(f"🚨 Model file not found at {model_path}. Please run `main.py` first!")

# # ✅ Initialize and Load MLP Model
# mlp_model = MLPClassifier(input_dim, hidden_dim1, hidden_dim2, output_dim, dropout_prob).to(device)
# mlp_model.load_state_dict(torch.load(model_path, map_location=device))
# mlp_model.eval()

# print("✅ Model loaded successfully for prediction.")

# # ✅ Define Emotion Labels
# emotion_labels = {0: "Angry", 1: "Sad", 2: "Neutral", 3: "Happy"}

# def predict_emotion(text):
#     encoding = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
#     input_ids = encoding["input_ids"].to(device)
#     attention_mask = encoding["attention_mask"].to(device)

#     with torch.no_grad():
#         outputs = indicbert_model(input_ids=input_ids, attention_mask=attention_mask)
#         hidden_states = outputs.hidden_states
#         sentence_embedding = hidden_states[-2][:, 0, :]  # Use CLS token for consistency

#         emotion_logits = mlp_model(sentence_embedding)
#         probabilities = F.softmax(emotion_logits, dim=1)
#         predicted_label = torch.argmax(probabilities, dim=1).item()

#     return emotion_labels[predicted_label]

# # ✅ Test Example (Optional)
# if __name__ == "__main__":
#     user_input = input("Enter a sentence: ")
#     emotion = predict_emotion(user_input)
#     print(f"Predicted Emotion: {emotion}")

import sys
import os
import torch
from transformers import AutoModel, AutoTokenizer

# ✅ Ensure 'model' folder is recognized
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ✅ Import MLPClassifier correctly
from model.model import MLPClassifier  # Import from model.py

# ✅ Define Model and Tokenizer
MODEL_NAME = "ai4bharat/indic-bert"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Load IndicBERT Model
indicbert_model = AutoModel.from_pretrained(MODEL_NAME, output_hidden_states=True).to(device)
indicbert_model.eval()

# ✅ Load Trained MLP Model
input_dim = indicbert_model.config.hidden_size  # 768 for IndicBERT
hidden_dim1 = 256
hidden_dim2 = 128
output_dim = 4
dropout_prob = 0.3

# ✅ Ensure correct model path
model_path = os.path.join(os.path.dirname(__file__), "mlp_model.pth")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"🚨 Model file not found at {model_path}. Please run `save_model.py` first!")

# ✅ Initialize and Load MLP Model
mlp_model = MLPClassifier(input_dim, hidden_dim1, hidden_dim2, output_dim, dropout_prob).to(device)
mlp_model.load_state_dict(torch.load(model_path, map_location=device))
mlp_model.eval()

print("✅ Model loaded successfully for prediction.")

# ✅ Define Emotion Labels
emotion_labels = {0: "Angry", 1: "Sad", 2: "Neutral", 3: "Happy"}

# ✅ Function to Predict Emotion
def predict_emotion(text):
    encoding = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = indicbert_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states
        second_to_last_layer = hidden_states[-2]
        sentence_embedding = second_to_last_layer[:, 0, :]

        emotion_logits = mlp_model(sentence_embedding)
        predicted_label = torch.argmax(emotion_logits, dim=1).item()

    return emotion_labels[predicted_label]

# ✅ Test Example (Optional)
if __name__ == "__main__":
    user_input = input("Enter a sentence: ")
    emotion = predict_emotion(user_input)
    print(f"Predicted Emotion: {emotion}")

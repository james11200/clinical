import torch
from transformers import AutoTokenizer, LlamaForSequenceClassification
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
import json
from datasets import Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# Load data from JSON files
def load_data(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file.readlines()]
    texts = [d['Input'].replace('\n', '').replace('\r', '') for d in data]
    labels = [d['Output'].replace('\n', '').replace('\r', '') for d in data]
    return texts, labels

# Convert label string to multi-label format
def convert_labels(label_str):
    label_map = {
        "Cardiac Arrest": 0,
        "ICU transfer": 1,
        "Inotropic Support": 2,
        "Mechanical Ventilation": 3
    }
    labels = [0.0] * len(label_map)
    for label in label_str.split(','):
        label_name, percentage = label.split('=')
        label_name = label_name.strip()
        percentage_value = float(percentage.strip().replace('%', '')) / 100
        if label_name in label_map:
            labels[label_map[label_name]] = percentage_value
    return labels

# Load test dataset
test_data = 'test_dataset_perc_ver2.json'  # Define the test data file name
test_texts, test_labels = load_data(test_data)
test_labels = [convert_labels(label) for label in test_labels]

# Load fine-tuned model and tokenizer
model_path = './fine-tuned-llama3-model(1)'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = LlamaForSequenceClassification.from_pretrained(
    model_path,
    num_labels=4,
    problem_type="multi_label_classification",
)

# Set the model to evaluation mode and move to GPU if available
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Function to make predictions for the entire test dataset
def predict_all(texts):
    all_probabilities = []
    for text in tqdm(texts, desc="Predicting"):
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        probabilities = torch.sigmoid(outputs.logits).cpu().numpy().flatten()
        all_probabilities.append(probabilities)
    return np.array(all_probabilities)

# Function to calculate ROC-AUC
def calculate_roc_auc(y_true, y_pred):
    roc_auc_scores = {}
    num_labels = y_true.shape[1]
    for i in tqdm(range(num_labels), desc="Calculating AUC"):
        roc_auc = roc_auc_score(y_true[:, i], y_pred[:, i])
        roc_auc_scores[f'Label {i}'] = roc_auc
    return roc_auc_scores

# Make predictions for test set
predictions = predict_all(test_texts)

# Calculate ROC-AUC
roc_auc_scores = calculate_roc_auc(np.array(test_labels), predictions)

# AUC Calculation and Saving to File
all_labels = np.array(test_labels)
all_preds = np.array(predictions)

# Calculate AUCs for each label
aucs = []
for i in range(all_labels.shape[1]):  # Iterate over each label
    auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
    aucs.append(auc)

# Print individual AUCs and the average AUC
print(f"AUCs for each label: {aucs}")
print(f"Average AUC: {np.mean(aucs)}")

# Save the AUC result to a file
model_name = 'fine-tuned-llama3-model(1)'  # Define the model name
with open('auc_result.txt', 'w', encoding='utf-8') as f:
    f.write(f"Model: {model_name}\n")
    f.write(f"Data name: {test_data}\n")
    f.write(f"AUCs for each label: {aucs}\n")
    f.write(f"Average AUC: {np.mean(aucs)}\n")

# Plot ROC Curves for each label and save images
for i in range(all_labels.shape[1]):  # Iterate over each label
    plt.figure(figsize=(10, 8))
    fpr, tpr, _ = roc_curve(all_labels[:, i], all_preds[:, i])
    plt.plot(fpr, tpr, label=f'Label {i} (AUC = {aucs[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.title(f'ROC Curve for {model_name} - Label {i}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid()

    # Save ROC curve image
    save_path = f'roc_curve_label_{i}_auc_{aucs[i]:.2f}.png'
    plt.savefig(save_path)
    print(f"ROC curve saved at: {save_path}")
    plt.close()

print("ROC curves and AUC results saved successfully.")

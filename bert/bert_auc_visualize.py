import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from sklearn.metrics import roc_auc_score, roc_curve
from datasets import Dataset
import json

# Load the fine-tuned models and tokenizers
model1 = "bvanaken/clinical-assertion-negation-bert"

model = "medicalai/ClinicalBERT"

models = [model1,model,f"./fine-tuned-{model.split('/')[-1]}"]#[model1, model2]
model_names = [model.split('/')[-1] for model in models]


def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4, ignore_mismatched_sizes=True)

    # model = AutoModelForSequenceClassification.from_pretrained(
    #     model_name, 
    #     num_labels=4, 
    #     ignore_mismatched_sizes=True  # Ignore size mismatch for classifier weights
    # )
    return tokenizer, model

def load_data(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file.readlines()]
    texts = [d['Input'].replace('\n', '').replace('\r', '') for d in data]
    labels = [d['Output'].replace('\n', '').replace('\r', '') for d in data]
    return texts, labels

def convert_labels(label_str):
    label_map = {
        "Cardiac Arrest": 0,
        "ICU transfer": 1,
        "Inotropic Support": 2,
        "Mechanical Ventilation": 3
    }
    labels = [0.0] * len(label_map)  # Initialize with 0.0 for all labels
    for label in label_str.split(','):
        label_name, percentage = label.split('=')  # Split label and percentage
        label_name = label_name.strip()  # Remove extra spaces
        percentage_value = float(percentage.strip().replace('%', '')) / 100  # Convert percentage to float

        if label_name in label_map:
            labels[label_map[label_name]] = percentage_value
    return labels

def create_dataset(texts, labels):
    labels = [torch.tensor(label, dtype=torch.float) for label in labels]
    return Dataset.from_dict({
        'text': texts,
        'labels': labels
    })

# Load the test data
test_texts, test_labels = load_data('test_dataset_perc_ver2.json')
test_labels = [convert_labels(label) for label in test_labels]
test_dataset = create_dataset(test_texts, test_labels)

# Tokenize the test dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)

# Store results for both models
results = {}

for model in models:
    tokenizer, model_instance = load_model(model)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    trainer = Trainer(
        model=model_instance,
        tokenizer=tokenizer
    )

    # Make predictions
    predictions = trainer.predict(test_dataset)
    logits = predictions.predictions
    labels = predictions.label_ids

    # Apply sigmoid to get probabilities
    sigmoid_logits = torch.sigmoid(torch.tensor(logits))
    preds = sigmoid_logits.cpu().numpy()

    # Calculate AUC for each label
    aucs = []
    for i in range(labels.shape[1]):
        auc = roc_auc_score(labels[:, i], preds[:, i])
        aucs.append(auc)

    results[model] = {
        'aucs': aucs,
        'preds': preds,
        'labels': labels
    }

    print(f"AUCs for {model}: {aucs}")

# Visualize ROC curves and save images
for i in range(len(models)):
    model = models[i]
    aucs = results[model]['aucs']
    preds = results[model]['preds']
    labels = results[model]['labels']

    plt.figure(figsize=(10, 8))
    mean_auc = np.mean(aucs)

    for j in range(labels.shape[1]):
        fpr, tpr, thresholds = roc_curve(labels[:, j], preds[:, j])
        plt.plot(fpr, tpr, label=f'Label {j} (AUC = {aucs[j]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.title(f'ROC Curves for {model}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid()
    # plt.savefig(f'roc_curve_{model_names[i]}_avg_auc_{mean_auc:.2f}(1).png')
    import os
    save_path = os.path.join(os.getcwd(), f'roc_curve_{model_names[i]}_avg_auc_{mean_auc:.2f}(1).png')
    plt.savefig(save_path)    
    plt.close()

# print(f"Average AUC for {model1}: {np.mean(results[model1]['aucs']):.2f}")
for model in models:
    print(f"Average AUC for {model}: {np.mean(results[model]['aucs']):.2f}")

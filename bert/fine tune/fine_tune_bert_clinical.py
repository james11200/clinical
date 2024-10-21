import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# Check if GPU is available
print(f'using GPU: {torch.cuda.is_available()}')

# Load data from JSON files
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
    labels = [0.0] * len(label_map)  # Use float
    # for label in label_str.split(','):
    #     label_name = label.split('=')[0].strip()
    #     if label_name in label_map:
    #         labels[label_map[label_name]] = 1.0
    for label in label_str.split(','):
        label_name, percentage = label.split('=')
        label_name = label_name.strip()
        percentage_value = float(percentage.strip().replace('%',''))/100
        if label_name in label_map:
            labels[label_map[label_name]] = percentage_value
    # print(labels)
    return labels


train_texts, train_labels = load_data('instruction_dataset_perc_ver2.json')
val_texts, val_labels = load_data('validation_dataset_perc_ver2.json')
test_texts, test_labels = load_data('test_dataset_perc_ver2.json')

# for t in train_texts:
#     print(len(t))
# breakpoint()

# Convert labels to multi-label format
train_labels = [convert_labels(label) for label in train_labels]
val_labels = [convert_labels(label) for label in val_labels]
test_labels = [convert_labels(label) for label in test_labels]

# Load the tokenizer
model1="bvanaken/clinical-assertion-negation-bert"
model2="medicalai/ClinicalBERT"
model=model2
model_name=model.split('/')[-1]
print(f'model name: {model_name}')

tokenizer = AutoTokenizer.from_pretrained(model)

# Tokenize function
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)

# Convert data to Dataset format
def create_dataset(texts, labels):
    # Convert labels to float tensors
    labels = [torch.tensor(label, dtype=torch.float) for label in labels]
    return Dataset.from_dict({
        'text': texts,
        'labels': labels
    })


# Create datasets
train_dataset = create_dataset(train_texts, train_labels)
val_dataset = create_dataset(val_texts, val_labels)
test_dataset = create_dataset(test_texts, test_labels)

# Tokenize datasets
train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Load the model
model = AutoModelForSequenceClassification.from_pretrained("medicalai/ClinicalBERT", num_labels=4, problem_type="multi_label_classification")

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs",
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

# Start training
trainer.train()

# Evaluate on test dataset
results = trainer.evaluate(test_dataset)
print(results)

# Save the fine-tuned model
model.save_pretrained(f'./fine-tuned-{model_name}')
tokenizer.save_pretrained(f'./fine-tuned-{model_name}')
# model.save_pretrained('./fine-tuned-clinical-assertion-negation-bert')
# tokenizer.save_pretrained('./fine-tuned-clinical-assertion-negation-bert')

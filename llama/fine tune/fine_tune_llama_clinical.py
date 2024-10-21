import json
import torch
from transformers import AutoTokenizer, LlamaForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from torch.cuda.amp import GradScaler

scaler = GradScaler()

# Check if GPU is available
print(f'using GPU: {torch.cuda.is_available()}')
torch.cuda.empty_cache()

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

# Load train, val, and test datasets
train_texts, train_labels = load_data('instruction_dataset_perc_ver2.json')
val_texts, val_labels = load_data('validation_dataset_perc_ver2.json')
test_texts, test_labels = load_data('test_dataset_perc_ver2.json')

# Convert labels to multi-label format
train_labels = [convert_labels(label) for label in train_labels]
val_labels = [convert_labels(label) for label in val_labels]
test_labels = [convert_labels(label) for label in test_labels]

# Load the LLaMA tokenizer and model for sequence classification
HF_API_TOKEN = "your api token"
llama_model = 'meta-llama/Meta-Llama-3-8B'

# Define LoRA configuration
lora_config = LoraConfig(
    r=2,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
)

# Load and apply LoRA to the model
model = LlamaForSequenceClassification.from_pretrained(
    llama_model,
    num_labels=4,
    problem_type="multi_label_classification",
    use_auth_token=HF_API_TOKEN
)
model = get_peft_model(model, lora_config)
model.config.use_cache=False
model.gradient_checkpointing_enable()

# for parameter in model.parameters():
#     parameter.requires_grad=True
for name, param in model.named_parameters():
    if not param.requires_grad:
        print(f"{name}: requires_grad=False")
    if "lora" in name:  # Ensure only LoRA layers are trainable
        param.requires_grad = True
    else:
        param.requires_grad = False  # Freeze other parameters

tokenizer = AutoTokenizer.from_pretrained(llama_model, use_auth_token=HF_API_TOKEN)

# Ensure the tokenizer has a pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
# Tokenize function
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)

# Convert data to Dataset format
def create_dataset(texts, labels):
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

# Custom Trainer to handle multi-label classification
class CustomTrainer(Trainer):

    # def compute_loss(self, model, inputs, return_outputs=False):
    #     # input['label'].require_grad = True
    #     outputs = model(**inputs)
    #     logits = outputs.logits
    #     labels = inputs.get("labels")
    #     print(f"Logits require grad: {logits.requires_grad}")
    #     print(f"Labels require grad: {labels.requires_grad}")  # Should be False
    #     # Use BCEWithLogitsLoss for multi-label classification
    #     loss_fct = torch.nn.BCEWithLogitsLoss()
    #     loss = loss_fct(logits, labels)
        
    #     return (loss, outputs) if return_outputs else loss

    def compute_loss(self, model, inputs, return_outputs=False):
        # Ensure input labels are tensors without requires_grad=True
        labels = inputs.get("labels").float()
        outputs = model(**inputs)

        # Ensure logits require grad
        logits = outputs.logits
        if not logits.requires_grad:
            logits.requires_grad = True
        # print(f"Logits require grad: {logits.requires_grad}")
        # print(f"Labels require grad: {labels.requires_grad}")     
        # Use BCEWithLogitsLoss for multi-label classification
        loss_fct = torch.nn.BCEWithLogitsLoss()
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss
# Define training arguments
training_args = TrainingArguments(
    fp16=False,
    output_dir="./results",
    evaluation_strategy="steps",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    logging_dir='logs',
    logging_steps=1000,
    # save_steps=2000,
    eval_steps=1000,
    num_train_epochs=3,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True
)

# Create Trainer instance with the custom loss computation
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Start training
trainer.train()

# Evaluate on test dataset
results = trainer.evaluate(test_dataset)
print(results)

# Save the fine-tuned model
model.save_pretrained('./fine-tuned-llama3-model(1)')
tokenizer.save_pretrained('./fine-tuned-llama3-model(1)')

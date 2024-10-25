import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from sklearn.metrics import roc_auc_score
from datasets import Dataset
import seaborn as sns
import json

# Model definitions
model_name = "medicalai/ClinicalBERT"
fine_tuned_model = f"./fine-tuned-{model_name.split('/')[-1]}"
# models = [model_name, fine_tuned_model]
models = [fine_tuned_model]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

def split_into_sentences(text):
    return sent_tokenize(text)

def map_tokens_to_sentences(text, tokenizer):
    sentences = split_into_sentences(text)
    tokenized = tokenizer.encode_plus(
        text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=512
    )
    tokens = tokenizer.convert_ids_to_tokens(tokenized['input_ids'])
    offsets = tokenized['offset_mapping']
    
    token_sentence_map = []
    current_sentence = 0
    current_char = 0
    for offset in offsets:
        if offset[0] == 0 and current_char < len(sentences[current_sentence]):
            # New sentence detected
            current_sentence += 1
            if current_sentence >= len(sentences):
                current_sentence = len(sentences) - 1
        token_sentence_map.append(current_sentence)
        current_char = offset[1]
    
    return tokens, token_sentence_map, sentences
def aggregate_attention_by_sentence(attention_weights, token_sentence_map, num_sentences):
    sentence_attention = np.zeros(num_sentences)
    token_counts = np.zeros(num_sentences)
    
    for i, sentence_id in enumerate(token_sentence_map):
        sentence_attention[sentence_id] += attention_weights[i]
        token_counts[sentence_id] += 1
    
    # Avoid division by zero
    token_counts[token_counts == 0] = 1
    sentence_attention /= token_counts
    
    return sentence_attention

def visualize_sentence_attention(sentence_attention, sentences):
    plt.figure(figsize=(12, 8))
    sns.barplot(x=sentence_attention, y=sentences, palette='viridis')
    plt.xlabel('Average Attention Weight')
    plt.ylabel('Sentences')
    plt.title('Sentence-Level Attention Weights')
    plt.tight_layout()
    plt.show()


# Load model and tokenizer
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=4, 
        output_attentions=True
    )
    return tokenizer, model

# Load dataset from JSON
def load_data(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file.readlines()]
    texts = [d['Input'].replace('\n', '').replace('\r', '') for d in data]
    labels = [d['Output'].replace('\n', '').replace('\r', '') for d in data]
    return texts, labels

# Convert percentage-based labels to numerical format
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

# Create dataset for PyTorch
def create_dataset(texts, labels):
    labels = [torch.tensor(label, dtype=torch.float) for label in labels]
    return Dataset.from_dict({'text': texts, 'labels': labels})

# Tokenization function
# def tokenize_function(examples): #tokenize for token
#     return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)

def tokenize_function(examples): #tokenize for sentence
    tokens_list = []
    sentence_maps = []
    sentences_list = []
    for text in examples['text']:
        tokens, sentence_map, sentences = map_tokens_to_sentences(text, tokenizer)
        tokens_list.append(tokens)
        sentence_maps.append(sentence_map)
        sentences_list.append(sentences)
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512), {
        'sentence_map': sentence_maps,
        'sentences': sentences_list
    }


# Extract attention weights and logits from model predictions
def extract_attention_weights(model, dataloader, device):
    model.eval()
    attention_data = []

    for batch in dataloader:
        inputs = {key: value.to(device) for key, value in batch.items() if key != 'labels'}

        with torch.no_grad():
            outputs = model(**inputs)
            attentions = outputs.attentions  # Extract attention weights
            
            # Collect the average attention weights across heads
            avg_attention = torch.mean(attentions[-1], dim=1).cpu().numpy()  # Use the last layer
            attention_data.append(avg_attention)
    
    return attention_data

# Load the test data
test_texts, test_labels = load_data('test_dataset_perc_ver2.json')
test_labels = [convert_labels(label) for label in test_labels]
test_dataset = create_dataset(test_texts, test_labels)

# Results storage
results = {}

# Iterate over each model
for model_path in models:
    tokenizer, model_instance = load_model(model_path)
    
    # Map the dataset with tokenization
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    
    # Initialize Trainer
    trainer = Trainer(model=model_instance, tokenizer=tokenizer)
    
    # Create dataloader
    dataloader = trainer.get_test_dataloader(test_dataset)
    
    # Move model to device
    model_instance.to(device)

    # Extract predictions and attention weights
    attention_data = extract_attention_weights(model_instance, dataloader, device)

    # Collect inputs and logits
    all_preds = []
    all_labels = []
    
    for batch in dataloader:
        inputs = {key: value.to(device) for key, value in batch.items() if key != 'labels'}
        labels = batch['labels'].to(device)

        with torch.no_grad():
            outputs = model_instance(**inputs)
            logits = outputs.logits

        all_preds.append(logits.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    
    preds = np.concatenate(all_preds, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    # Apply sigmoid to logits for probabilities (if required)
    sigmoid_logits = torch.sigmoid(torch.tensor(preds))
    preds = sigmoid_logits.cpu().numpy()

    # Calculate AUC for each label
    aucs = [roc_auc_score(labels[:, i], preds[:, i]) for i in range(labels.shape[1])]

    # Store results
    results[model_path] = {
        'aucs': aucs,
        'preds': preds,
        'labels': labels,
        'attentions': attention_data
    }

    print(f"AUCs for {model_path}: {aucs}")
    attentions=attention_data
    # for layer_idx, layer_attentions in enumerate(attentions):
    #     print(f"Attention scores for layer {layer_idx}:")
    #     print(layer_attentions)

    # Optionally: Aggregate attention scores across heads and layers for each token
    # Example: Averaging attention across all heads in a specific layer
    attention_layer_0 = attentions[0]  # First layer attention
    # If attention_layer_0 is a NumPy array, convert it to a PyTorch tensor
    if isinstance(attention_layer_0, np.ndarray):
        attention_layer_0 = torch.tensor(attention_layer_0)

    # Now compute the mean across the heads (dim=1)
    attention_avg = torch.mean(attention_layer_0, dim=1)
    
    # attention_avg = torch.mean(attention_layer_0, dim=1)  # Average across heads
    print(f"Average attention for first layer: {attention_avg}")
    
def pad_attention_weights(attention_weights, max_length):
    """
    Pad or trim the attention weights to ensure they are all max_length x max_length.
    Handle cases where layers have different numbers of heads.
    """
    padded_attention_weights = []

    # Iterate through each layer of attention weights
    for layer_attention in attention_weights:
        num_heads, current_len, _ = layer_attention.shape

        # Initialize a padded layer with all zeros (shape: max_heads x max_length x max_length)
        # We will pad missing heads if necessary
        padded_layer_attention = np.zeros((8, max_length, max_length))  # Assuming 8 heads max
        
        # If current_len is smaller than max_length, pad with zeros
        for head in range(num_heads):
            if current_len < max_length:
                padded_layer_attention[head, :current_len, :current_len] = layer_attention[head]
            else:
                padded_layer_attention[head] = layer_attention[head, :max_length, :max_length]

        # Handle cases where fewer than 8 heads are present by zero-padding the remaining heads
        for head in range(num_heads, 8):  # If there are fewer than 8 heads
            padded_layer_attention[head] = np.zeros((max_length, max_length))

        # Append the padded layer attention to the final list
        padded_attention_weights.append(padded_layer_attention)

    return padded_attention_weights





def get_token_word(token_id):
    """Get the word corresponding to the token ID from the vocabulary."""
    return id_to_token.get(str(token_id), '[UNK]')  # Ensure token_id is a string for consistency



def visualize_attention_weights(attention_weights, tokens, max_length=512):

    # Ensure all attention weights are padded to the same length
    if max_length is None:
        max_length = max([len(a) for a in attention_weights])
    
    attention_weights_padded = pad_attention_weights(attention_weights, max_length=max_length)
    
    # Average across heads
    avg_attention = np.mean(attention_weights_padded[0], axis=0)  # Averaging over heads
    plt.rc('font', family='Microsoft JhengHei')
    # Limit tokens to display up to max_length for visualization
    
    tokens_to_display = tokens[:max_length]  # Truncate tokens to max_length

    plt.figure(figsize=(max(10, len(tokens_to_display) * 0.2), max(10, len(tokens_to_display) * 0.2)))
    plt.imshow(avg_attention, cmap='viridis', aspect='auto')
    plt.colorbar(label='Attention Weight')

    # plt.xticks(ticks=np.arange(len(tokens_to_display)), labels=tokens_to_display, fontsize=6, rotation=90)
    # plt.yticks(ticks=np.arange(len(tokens_to_display)), labels=tokens_to_display, fontsize=6)
    
    plt.xticks([])
    plt.yticks([])
    
    step = 10  # Display every 10th token (adjust this value as needed)
    tokens_sampled = tokens[:max_length:step]  # Limit tokens to max_length, then sample every nth token
    plt.xticks(ticks=np.arange(0, len(tokens_sampled) * step, step), labels=tokens_sampled, fontsize=6, rotation=90)
    plt.yticks(ticks=np.arange(0, len(tokens_sampled) * step, step), labels=tokens_sampled, fontsize=6)
    
    
    plt.title("Average Attention Weights")
    plt.tight_layout()
    plt.savefig('attention_weights3.png')
    
    plt.show()



#Extract Attention Percentage for Each Token
def compute_attention_percentages(attention_matrices, texts):
    attention_percentages = []
    for head_idx, head_attention in enumerate(attention_matrices):
        if head_attention.shape == (8, 512, 512):  # Adjust this if needed
            # Process the attention matrix for this head
            print(f"Processing attention matrix for head {head_idx} with shape {head_attention.shape}")
            summed_attention = np.mean(head_attention, axis=1)  # Example: summing over heads
            attention_percentages.append(summed_attention)
        else:
            print(f"Skipping attention matrix for head {head_idx} due to shape mismatch: {head_attention.shape}")
    
    # Convert list of matrices into a NumPy array
    return np.array(attention_percentages)



# Load the tokenizer vocab from the JSON file
with open('fine-tuned-ClinicalBERT/tokenizer.json', 'r', encoding='utf-8') as f:
    tokenizer_data = json.load(f)
    vocab = tokenizer_data['model']['vocab']

# Reverse the vocabulary to map token IDs to words
id_to_token = {v: k for k, v in vocab.items()}

def get_token_word(token_id):
    """Get the word corresponding to the token ID from the vocabulary."""
    return id_to_token.get(token_id, '[UNK]')  # Use [UNK] if the token ID is not found in the vocabulary


# Example usage for visualizing attention weights
for model_path in models:
    if results[model_path]['attentions']:
        for i, text in enumerate(test_texts):
            tokens, sentence_map, sentences = map_tokens_to_sentences(text, tokenizer)
            attention_weights = results[model_path]['attentions'][i]  # Adjust indexing as needed
            sentence_attention = aggregate_attention_by_sentence(attention_weights, sentence_map, len(sentences))
            visualize_sentence_attention(sentence_attention, sentences)        
            '''
        # Visualize attention weights for this model
        visualize_attention_weights(results[model_path]['attentions'], test_texts)

        # Calculate and print attention percentages for this model
        attention_percentages = compute_attention_percentages(results[model_path]['attentions'], test_texts)
        # for i in range(len(test_texts)):
        #     for j in range(len(test_texts)):
        #         print(f"Token {i} -> Token {j}: {attention_percentages[i][j]}%")
        attention_data = []

        for i in range(attention_percentages.shape[1]):  # Limit 'i' to the valid range of axis 1
            for j in range(attention_percentages.shape[2]):  # Limit 'j' to the valid range of axis 2
                attention_value = attention_percentages[0, i, j] * 100  # Added [0] for the head
                # print(f"Raw attention value between token {i} and token {j}: {attention_value}")

                # If attention_value is greater than 0
                if attention_value > 0:
                    word_i = get_token_word(i)
                    word_j = get_token_word(j)
                    # print(f"Token '{word_i}' -> Token '{word_j}': {attention_value:.2f}%")
                    # Store the data in a dictionary format
                    attention_data.append({
                        'from_token': word_i,
                        'to_token': word_j,
                        'attention_value': round(attention_value, 2)
                    })                    
        # Sort the attention data by 'attention_value' in descending order
        attention_data_sorted = sorted(attention_data, key=lambda x: x['attention_value'], reverse=True)

        # Save the sorted data to a JSON file
        with open('focus_percentage.json', 'w', encoding='utf-8') as outfile:
            json.dump(attention_data_sorted, outfile, ensure_ascii=False, indent=4)
        print("Attention values saved to focus_percentage.json")

        # for i in range(attention_percentages.shape[1]):
        #     for j in range(attention_percentages.shape[2]):
        #         attention_value = attention_percentages[i, j] * 100

        #         print(f"Raw attention value between token {i} and token {j}: {attention_value}")

        #         # if attention_value > 0:
        #         if (attention_value > 0).all():

        #         #     print(f"Token {i} -> Token {j}: {attention_value:.2f}%")         
        #         # print(f"Token {i} -> Token {j}: {attention_value:.2f}%")         
        #             word_i = get_token_word(i)
        #             word_j = get_token_word(j)
        #             print(f"Token '{word_i}' -> Token '{word_j}': {attention_value:.2f}%")                    
        # for token, attention in attention_percentages:
        #     print(f"Token: {token}, Attention: {attention:.2f}%")
'''
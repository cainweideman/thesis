import torch
import torch.utils
from torch.utils.data import Dataset
from datasets import load_dataset, Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from sklearn.metrics import accuracy_score


def tokenize(examples):
    return tokenizer(
        examples['text'],
        add_special_tokens=True,
        truncation=True)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
	}

# Set seed
torch.manual_seed(42)

# If there's a GPU available...
if torch.cuda.is_available():    

    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

# Load datasets
dataset = load_dataset("GroNLP/dutch-cola")

# Remove unnecesary columns from the dataset
dataset = dataset.remove_columns(['Source', 'Original ID', 'Original annotation', 'Material added'])

# Rename the Sentence and Acceptability columns
dataset = dataset.rename_column("Sentence", "text")
dataset = dataset.rename_column("Acceptability", "label")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("yhavinga/gpt2-medium-dutch")
#tokenizer.padding_side = 'left'
tokenizer.pad_token = tokenizer.eos_token

# Load gpt2-small-dutch
model = AutoModelForSequenceClassification.from_pretrained("yhavinga/gpt2-medium-dutch", num_labels=2)
#model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = model.config.eos_token_id

# Tokenize the dataset
tokenized_dataset = dataset.map(tokenize, batched=True)

# Split the dataset
tokenized_train = tokenized_dataset['train']
tokenized_test = tokenized_dataset['test']
tokenized_valid = tokenized_dataset['validation']

# Create data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Tell PyTorch to run this model on GPU
model.to(device)

# Define training arguments
training_args = TrainingArguments(
    output_dir='D:/scriptie/gpt2-medium-dutch/output',
    logging_dir='logs',
    learning_rate=4e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    save_steps=124,
    eval_steps=124,
    logging_steps=124, 
    evaluation_strategy='steps',
    save_strategy='steps',
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model='eval_accuracy',
    fp16=True,
)

# Create trainer
trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_valid,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Evaluate model performance
trainer.evaluate(tokenized_test)

# Save the model
model.save_pretrained('D:/scriptie/gpt2-medium-dutch/fine-tuned-gpt2-medium')
tokenizer.save_pretrained('D:/scriptie/gpt2-medium-dutch/fine-tuned-gpt2-medium')
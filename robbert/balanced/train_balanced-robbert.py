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

# Read Balanced Dutch CoLa csv
train_df = pd.read_csv('D:/scriptie/balanced_cola_train.csv')

# Convert dataframe to dataset
train_data = Dataset.from_pandas(train_df)

# Remove unnecesary columns from the dataset
train_data = train_data.remove_columns(['Source', 'Original ID', 'Original annotation', 'Material added'])

# Rename the Sentence and Acceptability columns
train_data = train_data.rename_column("Sentence", "text")
train_data = train_data.rename_column("Acceptability", "label")

# Split the data
#train_data = dataset['train']
test_data = dataset['test']
valid_data = dataset['validation']

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("pdelobelle/robbert-v2-dutch-base")

# Tokenize the dataset
tokenized_dataset = dataset.map(tokenize, batched=True)

# Split the dataset
tokenized_train = train_data.map(tokenize, batched=True)
tokenized_test = tokenized_dataset['test']
tokenized_valid = tokenized_dataset['validation']

# Create data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Load RobBERT
model = AutoModelForSequenceClassification.from_pretrained("pdelobelle/robbert-v2-dutch-base", num_labels=2)

# Tell PyTorch to run this model on GPU
model.to(device)

# Define training arguments
training_args = TrainingArguments(
    output_dir='D:/scriptie/robbert/output',
    logging_dir='logs',
    learning_rate=5e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=2,
    save_steps=46,
    eval_steps=46,
    logging_steps=46,
    evaluation_strategy='steps',
    save_strategy='steps',
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
model.save_pretrained('D:/scriptie/balanced_robbert/fine-tuned-robbert')
tokenizer.save_pretrained('D:/scriptie/balanced_robbert/fine-tuned-robbert')
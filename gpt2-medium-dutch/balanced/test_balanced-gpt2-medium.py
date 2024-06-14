from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, DataCollatorWithPadding
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, matthews_corrcoef


def preprocess_function(examples):
	return tokenizer(examples['text'], add_special_tokens=True, truncation=True)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    matthews = matthews_corrcoef(labels, preds)
    return {
        'accuracy': acc,
        'matthews_corrcoef': matthews
	}

# Define model path
model_path = 'D:/scriptie/balanced_gpt2-medium/fine-tuned-gpt2-medium'

# Load fine-tuned gpt2-small-dutch model
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load BLIMP dataset
dataset = load_from_disk('D:/scriptie/test_dataset')

# Load data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Tokenize the dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Create trainer
trainer = Trainer(
	model=model,
	tokenizer=tokenizer,
	data_collator=data_collator,
	compute_metrics=compute_metrics
)

# Make list to save results
result_list = []

# Evaluate model with BLIMP datasets
for set in tokenized_dataset:
    evaluation_result = trainer.evaluate(tokenized_dataset[set])
    result_list.append(str(set) + ':\nAccuracy: ' + str(evaluation_result['eval_accuracy']) + '\nMatthews corrcoef: ' + str(evaluation_result['eval_matthews_corrcoef']))

new_result_list = [item.replace(".", ",") for item in result_list]

# Print results
for i in new_result_list:
     print(i)
     print('\n')
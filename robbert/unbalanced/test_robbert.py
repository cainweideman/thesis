from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, DataCollatorWithPadding
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef


def preprocess_function(examples):
	return tokenizer(examples['text'], truncation=True)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1_result = f1_score(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    matthews = matthews_corrcoef(labels, preds)
    return {
        'accuracy': acc,
        'matthews_corrcoef': matthews
	}

# Define model path
model_path = 'D:/scriptie/robbert/fine-tuned-robbert'

# Load fine-tuned RobBERT model
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Load RobBERT tokenizer
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

# Print results
for i in result_list:
     print(i)
     print('\n')
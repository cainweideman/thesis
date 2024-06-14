from transformers import TextClassificationPipeline, AutoModelForSequenceClassification, AutoTokenizer
from transformers.pipelines.pt_utils import KeyDataset
from datasets import Dataset, DatasetDict
import os
import pandas as pd
import pickle

directory = 'D:/scriptie/data/blimp/new_blimp'
data_dict = DatasetDict({})

for filename in os.listdir(directory):
	f = os.path.join(directory, filename)
	name = filename[:-4]
	
	df=pd.read_csv(f)
	dataset = Dataset.from_pandas(df)
	data_dict[name] = dataset

data_dict = data_dict.remove_columns(['Number (phenomenon)', 'Phenomenon', 'Number (paradigm)', 'Paradigm', 'Item', 'Critical word(s)', 'Cue word(s)'])
data_dict = data_dict.rename_column("Sentence", "text")
data_dict = data_dict.rename_column("Acceptability", "label")

tokenizer = AutoTokenizer.from_pretrained('D:/scriptie/gpt2-small-dutch/fine-tuned-gpt2-small')
model = AutoModelForSequenceClassification.from_pretrained('D:/scriptie/gpt2-small-dutch/fine-tuned-gpt2-small', label2id={'1': 1, '0': 0}, id2label={1: '1', 0: '0'})

pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, device=0)

prob_dict = {}
label_dict = {}

for i in data_dict:
	score_list = []
	label_list = []
	for out in pipe(KeyDataset(data_dict[i], 'text')):
		positive_probability = out['score'] if out['label'] == '1' else 1 - out['score']
		score_list.append(positive_probability)
		label_list.append(out['label'])

	prob_dict[i] = score_list
	label_dict[i] = label_list



with open("D:/scriptie/gpt2-small-dutch/probs/gpt2-small_scores.pkl", "wb") as outfile:
	pickle.dump(prob_dict, outfile)

with open("D:/scriptie/gpt2-small-dutch/probs/gpt2-small_labels.pkl", "wb") as f:
	pickle.dump(label_dict, f)

print('done')
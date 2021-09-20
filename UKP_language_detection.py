## pip install -r requirements.txt


from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, TrainingArguments, Trainer
import tensorflow as tf
import datasets
from datasets import load_dataset
import json
import torch
from torch import nn
from transformers import Trainer
from sklearn.model_selection import train_test_split

tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
model = XLMRobertaForSequenceClassification.from_pretrained('xlm-roberta-base', num_labels = 13)

def load_data (dir):
    with open(dir, encoding='utf-8') as j:
        data = json.load(j)
    return(data)

labels_dict = {'en':1, 'nl':2, 'bg':3, 'fa':4, 'da':5, 'et':6, 'fr':7, 'it':8, 'ko':9, 'zh':10, 'ja':11, 'hr':12, 'de':0}

train_texts = [d['text'] for d in load_data('training_data.json')]
train_labels =  [labels_dict[d['lang']] for d in load_data('training_data.json')]
val_texts =  [d['text'] for d in load_data('test_data.json')]
val_labels = [labels_dict[d['lang']] for d in load_data('test_data.json')]

data = ''

tokenized_datasets_train = tokenizer(train_texts, padding="max_length", truncation=True)
tokenized_datasets_val = tokenizer(val_texts, padding="max_length", truncation=True)

train_texts = []
val_texts =  []


class Lang_Dataset(torch.utils.data.Dataset):

    def __init__(self, encodings, labels):
        
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):

        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = Lang_Dataset(tokenized_datasets_train, train_labels)
val_dataset = Lang_Dataset(tokenized_datasets_val, val_labels)

tokenized_datasets_train = []
tokenized_datasets_val = []
train_labels =  []
val_labels = []

training_args = TrainingArguments(do_eval=True, do_train=True, num_train_epochs = 3.0, save_strategy ='steps', 
                                  output_dir = 'Result', evaluation_strategy ='steps', eval_steps = 50, save_steps = 2000, 
                                  gradient_accumulation_steps = 1, save_total_limit = 2,load_best_model_at_end=True)
trainer = Trainer(model=model, args = training_args, train_dataset=train_dataset, eval_dataset = val_dataset)
finetune_model = trainer.train()
trainer.save_model('Result/fine_tuned')

print(finetune_model)
metric = trainer.evaluate()
print(metric)


## ===========================================================================================================================================================================
"fine-tuned by using colab-gpu"

## ===========================================================================================================================================================================
'EXAMPLES you can take '

model_finetuned = XLMRobertaForSequenceClassification.from_pretrained(r'Result/fine_tuned')
prediction_dict = {1:'en', 2:'nl', 3:'bg', 4:'fa', 5:'da', 6:'et', 7:'fr', 8:'it', 9:'ko', 10:'zh', 11:'ja', 12:'hr', 0:'de'}

inputs = [tokenizer('Es geht mir gut.', return_tensors = 'pt'),tokenizer('이게 뭐야?', return_tensors = 'pt'),tokenizer('Waar kom je vandaan?', return_tensors = 'pt')]

outputs = [model_finetuned(**i) for i in inputs]
prediction_scores = [[{"label": prediction_dict[c], "score": i} for c,i in enumerate (o.logits.softmax(dim=1).tolist()[0])] for o in outputs]


for i in prediction_scores:
    max = 0
    for result in i:
        if result['score'] > max:
            max = result['score']
            key = result['label']
    print('Predicted as '+ key)

print(prediction_scores[0])

model_finetuned
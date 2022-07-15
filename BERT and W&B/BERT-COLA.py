import transformers
import wget
import os
import zipfile
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import wandb
import random
import numpy as np
from transformers import get_linear_schedule_with_warmup
import time
import datetime

#os.makedirs('/Users/ehsan/.config/wandb')

temp1 = os.environ.get('WANDB_CONFIG_DIR')
temp2 = os.path.join(os.path.expanduser("~"), ".config", "wandb")
temp3 = 'WANDB_CONFIG_DIR', os.path.join(os.path.expanduser("~"), ".config", "wandb")
temp4 = os.environ.get('WANDB_CONFIG_DIR')
config_dir = os.environ.get('WANDB_CONFIG_DIR', os.path.join(os.path.expanduser("~"), ".config", "wandb"))



sweep_config = {
    'method': 'random', #grid, random
    'metric': {
      'name': 'val_accuracy',
      'goal': 'maximize'
    },
    'parameters': {
        'learning_rate': {
            'values': [ 5e-5, 3e-5, 2e-5]
        },
        'batch_size': {
            'values': [16, 32]
        },
        'epochs':{
            'values':[2, 3, 4]
        }
    }
}
#sweep_id = wandb.sweep(sweep_config)


print('Downloading dataset...')

# The URL for the dataset zip file.
file_name = 'cola_public_1.1.zip'
url = 'https://nyu-mll.github.io/CoLA/' + file_name

# Download the file (if we haven't already)
path_to_zip_file = './' + file_name
if not os.path.exists(path_to_zip_file):
    wget.download(url, path_to_zip_file)

path_to_data = './cola_public/'
if not os.path.exists(path_to_data):
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall('./')


df = pd.read_csv("./cola_public/raw/in_domain_train.tsv", delimiter='\t', header=None,
names =  ['sentence_source', 'label', 'label_notes', 'sentence'])

print('Number of training sentences: {:,}\n'.format(df.shape[0]))
df.sample(10)

print(df.loc[df.label==0].sample(5)[['sentence', 'label']])

sentences = df.sentence.values
labels = df.label.values


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

'''
max_len = 0
for sent in sentences:
    input_ids = tokenizer.encode(sent, add_special_tokens=True)
    max_len = max(max_len, len(input_ids))

print(max_len)
'''

input_ids = []
attention_masks = []

for sent in sentences:
    encoded_dict = tokenizer.encode_plus(sent, add_special_tokens=True, max_len=64, pad_to_max_length=True, return_attention_mask=True, return_tensors='pt')
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])

input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks)
labels = torch.tensor(labels)

dataset = TensorDataset(input_ids, attention_masks, labels)

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


def ret_dataloader():
    #batch_size = wandb.config.batch_size
    batch_size = 128
    print(f'Batch size: {batch_size}')
    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
    validation_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size)
    return train_dataloader, validation_dataloader

def ret_model():
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2, output_attentions = False, output_hidden_states = False)
    return model

def ret_optim(model):
    #print(f'Learning_rate = {wandb.config.learning_rate}')
    learning_rate = 0.002
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
    return optimizer

def ret_scheduler(train_dataloader, optimizer):
    #epochs = wandb.config.epochs
    epochs = 5
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_training_steps=total_steps, num_warmup_steps=0)
    return scheduler

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def train():
    #wandb.init()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = ret_model()
    model.to(device)
    train_dataloader, validation_dataloader = ret_dataloader()
    optimizer = ret_optim(model)
    scheduler = ret_scheduler(train_dataloader, optimizer)
    training_stats = []
    total_t0 = time.time()
    #epochs = wandb.config.epochs
    epochs = 5
    for epoch_i in range(0, epochs):
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        t0 = time.time()
        total_train_loss = 0
        model.train()
        for step, batch in enumerate(train_dataloader):
            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            model.zero_grad()
            loss, logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            #wandb.log({'train_batch_loss':loss.item()})
            print(f'train_batch_loss: {loss.item()}')
            total_train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        avg_train_loss = total_train_loss / len(train_dataloader)
        training_time = format_time(time.time() - t0)
        #wandb.log({'avg_train_loss': avg_train_loss})
        print(f'avg_train_loss: {avg_train_loss}')
        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))
        print("Running Validation...")
        t0 = time.time()
        model.eval()
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0
        # Evaluate data for one epoch
        for batch in validation_dataloader:
            b_input_ids = batch[0].cuda()
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            with torch.no_grad():
                (loss, logits) = model(b_input_ids,
                                       token_type_ids=None,
                                       attention_mask=b_input_mask,
                                       labels=b_labels)

            total_eval_loss += loss.item()
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            total_eval_accuracy += flat_accuracy(logits, label_ids)

        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
        avg_val_loss = total_eval_loss / len(validation_dataloader)

        validation_time = format_time(time.time() - t0)
        # Log the Avg. validation accuracy
        #wandb.log({'val_accuracy': avg_val_accuracy, 'avg_val_loss': avg_val_loss})
        print("  Validation Loss: {0:.2f}".format(avg_val_loss))


train()
#wandb.agent(sweep_id, train)








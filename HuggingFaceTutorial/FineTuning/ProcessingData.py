from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader


raw_datasets = load_dataset('glue', 'mrpc')
print(raw_datasets)

raw_train_dataset = raw_datasets['train']
print(raw_train_dataset[0])
print(raw_train_dataset[:3])

print(raw_train_dataset.features)


checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

tokenized_sentences_1 = tokenizer(raw_datasets['train']['sentence1'], max_length=12, truncation=True)
tokenized_sentences_2 = tokenizer(raw_datasets['train']['sentence2'])
print(tokenized_sentences_1)


inputs = tokenizer("This is the first sentence.", "This is the second one.")
print(inputs)
decoded = tokenizer.convert_ids_to_tokens(inputs['input_ids'])
print(decoded)


tokenized_dataset = tokenizer(raw_datasets['train']['sentence1'], raw_datasets['train']['sentence2'],
                              padding=True, truncation=True)

def tokinize_function(example):
    #padding='max_length', max_length=128
    return tokenizer(example['sentence1'], example['sentence2'], truncation=True)

tokenized_dataset = raw_datasets.map(tokinize_function, batched=True)
print(tokenized_dataset.column_names)
print(tokenized_dataset)



data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

samples = tokenized_dataset['train'][:8]
print(samples)
samples = {k: v for k, v in samples.items() if k not in ['idx', 'sentence1', 'sentence2']}
print([len(x) for x in samples['input_ids']])


tokenized_dataset = tokenized_dataset.remove_columns(['idx', 'sentence1', 'sentence2'])
tokenized_dataset = tokenized_dataset.rename_column('label', 'labels')
tokenized_dataset = tokenized_dataset.with_format('torch')

train_dataloader = DataLoader(tokenized_dataset['train'], batch_size=16, shuffle=True, collate_fn=data_collator)

for step, batch in enumerate(train_dataloader):
    print(batch['input_ids'].shape)
    if step > 5:
        break





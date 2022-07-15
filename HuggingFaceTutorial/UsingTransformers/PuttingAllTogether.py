import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = [
  "I've been waiting for a HuggingFace course my whole life.",
  "So have I!"
]

tokens = tokenizer(sequences, padding=True, truncation=True, return_tensors='pt')
output = model(**tokens)
print(output)



tokens = tokenizer("Hello world")
print(tokens)
print(tokens['input_ids'])
tokens = tokenizer.tokenize('Hello world')
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)



l1 = [['a', 'b'], ['c', 'd']]
print(l1)
#print(**l1)
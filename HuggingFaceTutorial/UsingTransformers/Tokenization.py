from transformers import AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
sequence = "Using a Transformer network is simple"
tokens = tokenizer.tokenize(sequence)
print(tokens)
tokens = tokenizer.prepare_for_model(tokens)

ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)
decoded_string = tokenizer.decode(ids)
print(decoded_string)

ids = tokens['input_ids']
print(ids)




ids = tokenizer.convert_tokens_to_ids(tokens)
input_ids = torch.tensor([ids])
print(input_ids)






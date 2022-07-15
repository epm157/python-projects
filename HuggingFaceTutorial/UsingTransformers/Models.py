from transformers import BertConfig, BertModel
import torch

config = BertConfig()
model = BertModel(config)
print(model)


model = BertModel.from_pretrained('bert-base-cased')


model.save_pretrained('saved')

model = BertModel.from_pretrained('saved')


encoded_sequences = [
  [ 101, 7592,  999,  102],
  [ 101, 4658, 1012,  102],
  [ 101, 3835,  999,  102]
]


model_inputs = torch.tensor(encoded_sequences)

output = model(model_inputs)
print(output)









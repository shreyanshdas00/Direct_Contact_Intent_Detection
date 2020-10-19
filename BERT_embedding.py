#importing the pre-trained BERT embeddings model
import torch
from transformers import BertModel, BertTokenizer

class BERT():
  def __init__(self):
    PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'
    self.__tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    self.__model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)

  def __call__(self, utterances):
    input_ids = []
    attention_mask = []
    for utterance in utterances:
      encoding = self.__tokenizer.encode_plus(
      utterance,
      max_length=None,
      #truncation=True,
      add_special_tokens=True, # Add '[CLS]' and '[SEP]'
      return_token_type_ids=False,
      padding= max_length,
      return_attention_mask=True,
      return_tensors='pt',  # Return PyTorch tensors
      )
      input_ids.append(encoding['input_ids'])
      attention_mask.append(encoding['attention_mask'])
    input_ids = torch.cat(input_ids, 0)
    attention_mask = torch.cat(attention_mask, 0)
    last_hidden_state, pooled_output = self.__model(
    input_ids = input_ids,
    attention_mask = attention_mask
    )
    return last_hidden_state




#importing the pre-trained BERT embeddings model
import torch
from transformers import BertModel, BertTokenizer

class BERT():
  def __init__(self):
    PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'
    self.__tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    self.__model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)

  def __call__(self, utterance):
    encoding = self.__tokenizer.encode_plus(
    utterance,
    #max_length=512,
    #truncation=True,
    add_special_tokens=True, # Add '[CLS]' and '[SEP]'
    return_token_type_ids=False,
    #pad_to_max_length=True,
    return_attention_mask=True,
    return_tensors='pt',  # Return PyTorch tensors
    )
    last_hidden_state, pooled_output = self.__model(
    input_ids=encoding['input_ids'],
    attention_mask=encoding['attention_mask']
    )
    print(last_hidden_state.shape)




#importing the pre-trained BERT embeddings model
import torch
from transformers import BertModel, BertTokenizer

class BERT():
  def __init__():
    PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    bert_model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)

  def forward(utterance):
    encoding = tokenizer.encode_plus(
    sample_txt,
    #max_length=512,
    #truncation=True,
    add_special_tokens=True, # Add '[CLS]' and '[SEP]'
    return_token_type_ids=False,
    #pad_to_max_length=True,
    return_attention_mask=True,
    return_tensors='pt',  # Return PyTorch tensors
    )
    last_hidden_state, pooled_output = bert_model(
    input_ids=encoding['input_ids'],
    attention_mask=encoding['attention_mask']
    )
    print(last_hidden_state.shape)




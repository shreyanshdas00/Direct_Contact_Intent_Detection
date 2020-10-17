#importing the pre-trained BERT embeddings model
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

from transformers import BertModel, BertTokenizer
PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

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

bert_model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
last_hidden_state, pooled_output = bert_model(
  input_ids=encoding['input_ids'],
  attention_mask=encoding['attention_mask']
)
print(last_hidden_state.shape)

class BERT():
  def __init__(self, list_input):
    self.__tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    self.__model = AutoModel.from_pretrained("bert-base-uncased", output_hidden_states=True)

    self.__model.to('cuda')
    
    return self.get_embedding(list_input)


  #utility function to extract word embeddings for a given utterance
  def get_embedding(self, list_input):

    em_words=[]
    for question in list_input:
      ids=self.__tokenizer.encode(question)
      ids = torch.LongTensor(ids).unsqueeze(0)
      em_words.append(ids)
    embeddings=[]
    with torch.no_grad():
      for x in em_words:
        x=x.to('cuda')
        print(len(self.__model(input_ids=x)))
        embeddings.append((self.__model(input_ids=x)[2])[-1])
    return embeddings
 


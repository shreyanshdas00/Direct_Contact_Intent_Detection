#importing the pre-trained BERT embeddings model
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")

model.to('cuda')


#utility function to extract word embeddings for a given utterance
def get_chunk_embedding(list_input):
  
  em_cls=[]
  for question in list_input:
    ids=tokenizer.encode(question)
    ids = torch.LongTensor(ids).unsqueeze(0)
    em_cls.append(ids)
  sentence_embeddings=[]
  with torch.no_grad():
    for x in em_cls:
      x=x.to(device)
      sentence_embeddings.append((model(input_ids=x)[2])[-1])
  return sentence_embeddings


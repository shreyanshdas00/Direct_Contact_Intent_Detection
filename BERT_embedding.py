#importing the pre-trained BERT embeddings model
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")

model.to('cuda')


#utility function to extract word embeddings for a given utterance
def get_embedding(list_input):
  
  em_words=[]
  for question in list_input:
    ids=tokenizer.encode(question)
    ids = torch.LongTensor(ids).unsqueeze(0)
    em_words.append(ids)
  embeddings=[]
  with torch.no_grad():
    for x in em_words:
      x=x.to(device)
      embeddings.append((model(input_ids=x)[2])[-1])
  return embeddings


import torch
from mlp import ToxicityClassifier
from transformers import DistilBertTokenizer, DistilBertModel

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

model_path = "model_epoch_4001.pth"
checkpoint = torch.load(model_path)

classifier = ToxicityClassifier()
classifier.load_state_dict(checkpoint["model_state_dict"])
classifier.to(device)
classifier.eval()

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
distilbert_model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)

def predict(text):
    tokens = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
    input_ids = tokens['input_ids'].to(device)
    attention_mask = tokens['attention_mask'].to(device)
    with torch.no_grad():
        embeddings = distilbert_model(input_ids, attention_mask).last_hidden_state
        output = classifier(embeddings, attention_mask)
    return output

labels = ['severe_toxicity','obscene','identity_attack','insult','threat']

text = "People are stupid"
output = predict(text)
zipped = zip(labels, *output.tolist())

# print zipped
for k in zipped:
    print(k)

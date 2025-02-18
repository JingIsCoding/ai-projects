import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertModel
from mlp import ToxicityClassifier
from tqdm.auto import tqdm
import numpy as np

# get device if MPS is available
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
distilbert_model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)

def collate_fn(batch):
    texts, labels = zip(*batch)  # Unpack batch
    labels = np.array(labels)
    try:
        tokens = tokenizer(list(texts), padding=True, truncation=True, max_length=512, return_tensors="pt")
        labels = torch.from_numpy(labels).float().to(device)
    except Exception as e:
        print(f"Error: {e}")
        print(f"Texts: {texts}")
        return [], [], []
    return tokens['input_ids'].to(device), tokens['attention_mask'].to(device), labels.to(device)


class ToxicityDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        self.texts = df['comment_text'].tolist()
        self.labels = df[['severe_toxicity','obscene','identity_attack','insult','threat']].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        x = self.texts[idx]
        y = self.labels[idx]
        return x, y


dataset = ToxicityDataset('train.csv')
dataloader = DataLoader(dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)

classifier = ToxicityClassifier()
classifier.to(device)

optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-5)
criterion = torch.nn.BCELoss()

model_path = 'model_epoch_4001.pth'

if model_path:
    checkpoint = torch.load(model_path)
    classifier.load_state_dict(checkpoint["model_state_dict"])
    print(f"Model loaded from checkpoint {model_path}")

classifier.train()
distilbert_model.eval()

for epoch in range(5):
    steps = 0
    epoch_loss = 0.0
    for input_ids, attention_mask, labels in tqdm(dataloader):
        if len(input_ids) == 0:
            continue
        input_ids, attention_mask, labels = (
            input_ids,
            attention_mask,
            labels,
        )
        with torch.no_grad():
            outputs = distilbert_model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state
        output = classifier(embeddings, attention_mask)
        print(torch.mean(labels, dim=0))
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        steps += 1
        if steps % 1000 == 0:
            print(f"Epoch {epoch+1}, step {steps}: loss {loss.item()}")
            checkpoint_path = f"model_epoch_{steps+1}.pth"
            torch.save({
                "epoch": epoch+1,
                "model_state_dict": classifier.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss.item(),
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

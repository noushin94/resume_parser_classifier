import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, AdamW
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from app.config import Config
from app.data_preprocessing import preprocess_dataframe
from models.bert_classifier import BertClassifier  # Import the BertClassifier

class ResumeDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = ResumeDataset(
        texts=df['cleaned_text'].to_numpy(),
        labels=df['label'].to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(ds, batch_size=batch_size, num_workers=2)

def train_model():
    df = pd.read_csv(Config.DATA_PATH)
    df = preprocess_dataframe(df, 'Resume_str')

    # Label Encoding
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['Category'])

    # Save the label encoder classes for later use
    label_classes = le.classes_
    with open('models/label_classes.txt', 'w') as f:
        for item in label_classes:
            f.write("%s\n" % item)

    # Split Data
    X_train, X_val, y_train, y_val = train_test_split(
        df['cleaned_text'],
        df['label'],
        test_size=0.1,
        random_state=42,
        stratify=df['label']
    )

    # Tokenizer Initialization
    tokenizer = BertTokenizer.from_pretrained(Config.MODEL_NAME)

    # Model Initialization using BertClassifier
    num_labels = len(le.classes_)
    classifier = BertClassifier(model_name=Config.MODEL_NAME, num_labels=num_labels)

    # Data Loaders
    train_data_loader = create_data_loader(pd.DataFrame({'cleaned_text': X_train, 'label': y_train}),
                                           tokenizer, Config.MAX_LEN, Config.BATCH_SIZE)
    val_data_loader = create_data_loader(pd.DataFrame({'cleaned_text': X_val, 'label': y_val}),
                                         tokenizer, Config.MAX_LEN, Config.BATCH_SIZE)

    # Optimizer and Scheduler
    optimizer = AdamW(classifier.model.parameters(), lr=2e-5)

    # Training Loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classifier.to(device)

    for epoch in range(Config.EPOCHS):
        classifier.train()
        total_train_loss = 0

        for batch in train_data_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = classifier.forward(input_ids=input_ids,
                                         attention_mask=attention_mask,
                                         labels=labels)

            loss = outputs.loss
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_train_loss = total_train_loss / len(train_data_loader)
        print(f'Epoch {epoch+1}/{Config.EPOCHS}, Training Loss: {avg_train_loss:.4f}')

    # Save Model and Tokenizer
    classifier.save('models/saved_model')
    tokenizer.save_pretrained('models/saved_model')
    print('Model and tokenizer saved.')

if __name__ == '__main__':
    train_model()

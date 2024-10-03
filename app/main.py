from flask import Flask, request, render_template
from app.resume_parsing import parse_resume
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from app.config import Config
import os

app = Flask(__name__)

# Load the trained model and tokenizer
model = BertForSequenceClassification.from_pretrained('models/saved_model')
tokenizer = BertTokenizer.from_pretrained('models/saved_model')
model.eval()

# Load label encoder classes
label_classes = ['Data Scientist', 'Software Engineer', 'Project Manager']  

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_resume():
    if 'resume' not in request.files:
        return 'No file part'
    file = request.files['resume']
    if file.filename == '':
        return 'No selected file'
    if file:
        filepath = os.path.join('uploads/', file.filename)
        file.save(filepath)

        # Parse the resume
        parsed_data = parse_resume(filepath)

        # Classify the resume
        inputs = tokenizer(parsed_data['text'], return_tensors='pt', truncation=True, max_length=512)
        outputs = model(**inputs)
        _, prediction = torch.max(outputs.logits, dim=1)
        predicted_role = label_classes[prediction]

        return render_template('result.html', data=parsed_data, role=predicted_role)

if __name__ == '__main__':
    app.run(debug=True)

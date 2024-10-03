import torch
from transformers import BertForSequenceClassification

class BertClassifier:
    def __init__(self, model_name='bert-base-uncased', num_labels=2):
        """
        Initializes the BERT Classifier model.

        Args:
            model_name (str): Name of the pre-trained BERT model.
            num_labels (int): Number of output labels/classes.
        """
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )

    def to(self, device):
        """
        Moves the model to the specified device.

        Args:
            device (torch.device): The device to move the model to.
        """
        self.model.to(device)

    def train(self):
        """
        Sets the model in training mode.
        """
        self.model.train()

    def eval(self):
        """
        Sets the model in evaluation mode.
        """
        self.model.eval()

    def save(self, save_directory):
        """
        Saves the model and tokenizer to the specified directory.

        Args:
            save_directory (str): The directory to save the model.
        """
        self.model.save_pretrained(save_directory)

    def load(self, load_directory):
        """
        Loads the model from the specified directory.

        Args:
            load_directory (str): The directory to load the model from.
        """
        self.model = BertForSequenceClassification.from_pretrained(load_directory)

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Performs a forward pass through the model.

        Args:
            input_ids (torch.Tensor): Input IDs tensor.
            attention_mask (torch.Tensor): Attention mask tensor.
            labels (torch.Tensor, optional): Labels tensor.

        Returns:
            outputs: The output from the model.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs

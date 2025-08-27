import sys
import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BertForQuestionAnswering, BertTokenizerFast
from dataset_preparation import SquadDataset, load_squad, add_token_positions
import urllib.request

# Ensure parent directory is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def download_squad(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"📥 Downloading SQuAD v2.0 dataset to: {path}")
    url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json"
    urllib.request.urlretrieve(url, path)
    print("✅ Download complete.")

def train_for_steps(num_steps=500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    # Path to training dataset
    data_path = os.path.join("data", "squad_v2", "train-v2.0.json")
    if not os.path.exists(data_path):
        download_squad(data_path)

    # Load and preprocess dataset
    contexts, questions, answers, _ = load_squad(data_path)
    encodings = tokenizer(questions, contexts, truncation=True, padding=True, return_offsets_mapping=True)
    add_token_positions(encodings, answers)
    encodings.pop("offset_mapping")
    dataset = SquadDataset(encodings)

    # Create DataLoader
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Load model and optimizer
    model = BertForQuestionAnswering.from_pretrained("bert-base-uncased").to(device)
    optim = AdamW(model.parameters(), lr=5e-5)

    steps = 0
    total_loss = 0

    while steps < num_steps:
        model.train()
        batch = next(iter(loader))  # Get the next batch of data

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)

        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask,
                        start_positions=start_positions,
                        end_positions=end_positions)
        
        loss = outputs.loss
        total_loss += loss.item()

        # Backward pass
        loss.backward()

        # Update the weights
        optim.step()
        optim.zero_grad()

        steps += 1

        if steps % 10 == 0:  # Print progress every 10 steps
            avg_loss = total_loss / steps
            print(f"📊 Step {steps}/{num_steps} completed. Avg Loss: {avg_loss:.4f}")

    # Save the model after 500 steps
    final_dir = os.path.join("models", "final_model")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print("✅ Training complete. Final model saved at:", final_dir)

if __name__ == "__main__":
    train_for_steps(500)  # Run for 500 steps
import sys
import os
import torch
import json
from transformers import BertForQuestionAnswering, BertTokenizerFast

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.dataset_preparation import load_squad
from utils.squad_evaluation import compute_metrics

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer and model
    tokenizer = BertTokenizerFast.from_pretrained("models/final_model")
    model = BertForQuestionAnswering.from_pretrained("models/final_model").to(device)
    model.eval()

    # Load dev set
    data_path = os.path.join("data", "squad_v2", "dev-v2.0.json")
    if not os.path.exists(data_path):
        data_path = os.path.join("..", "data", "squad_v2", "dev-v2.0.json")
    
    contexts, questions, answers, _ = load_squad(data_path)
    
    # Only evaluate on the first 500 samples
    contexts = contexts[:500]
    questions = questions[:500]
    answers = answers[:500]
    
    # Tokenize the questions and contexts
    encodings = tokenizer(questions, contexts, truncation=True, padding=True, return_offsets_mapping=True, return_tensors="pt")

    predictions = {}
    with torch.no_grad():
        for i in range(len(questions)):
            input_ids = encodings['input_ids'][i].unsqueeze(0).to(device)
            attention_mask = encodings['attention_mask'][i].unsqueeze(0).to(device)

            # Get the model outputs
            outputs = model(input_ids, attention_mask=attention_mask)
            start = torch.argmax(outputs.start_logits)
            end = torch.argmax(outputs.end_logits) + 1

            # Get the predicted answer
            answer_tokens = input_ids[0][start:end]
            answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
            predictions[str(i)] = answer

    # Prepare ground truths
    ground_truths = {str(i): answers[i]['text'] for i in range(len(answers))}

    # Compute evaluation metrics
    metrics = compute_metrics(predictions, ground_truths)

    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/predictions.json", "w") as f:
        json.dump(predictions, f, indent=2)
    with open("results/evaluation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Print summary
    print(f"✅ Evaluation completed.")
    print(f"📊 Exact Match: {metrics['exact_match']:.2f}%")
    print(f"📊 F1 Score: {metrics['f1']:.2f}%")

if __name__ == "__main__":
    evaluate()
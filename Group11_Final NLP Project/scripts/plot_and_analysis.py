import json
import matplotlib.pyplot as plt
import os


def plot_training_loss(log_path="./results/training_log.json", output_path="./results/loss_plot.png"):
    # Load training loss log (from a JSON file, NOT an image!)
    with open(log_path, "r", encoding="utf-8") as f:
        log = json.load(f)

    epochs = list(range(1, len(log["losses"]) + 1))
    losses = log["losses"]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, losses, marker="o", label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.grid(True)
    plt.legend()
    plt.savefig(output_path)
    print(f"✅ Saved loss plot to {output_path}")


def generate_qualitative_analysis(
    pred_path="results/predictions.json",
    data_path="data/squad_v2/dev-v2.0.json",
    output_path="results/qualitative_analysis.txt",
    num_samples=10
):
    # Ensure the data path is correct, allow a fallback to project root
    if not os.path.exists(data_path):
        data_path = os.path.join("..", data_path)  # Check one directory level up
        if not os.path.exists(data_path):
            print(f"❌ Error: The data file at {data_path} was not found!")
            return

    with open(pred_path, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    with open(data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    examples = []
    for article in dataset["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                qid = qa["id"]
                question = qa["question"]
                gold_answers = [ans["text"] for ans in qa["answers"]]
                pred_answer = predictions.get(qid, "[No prediction]")
                examples.append({
                    "question": question,
                    "context": context,
                    "gold_answers": gold_answers,
                    "predicted_answer": pred_answer
                })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for i, ex in enumerate(examples[:num_samples]):
            f.write(f"Example {i+1}:\n")
            f.write(f"Q: {ex['question']}\n")
            f.write(f"A (Predicted): {ex['predicted_answer']}\n")
            f.write(f"A (Ground Truth): {ex['gold_answers']}\n")
            f.write(f"Context: {ex['context'][:300]}...\n")  # truncate long context
            f.write("=" * 80 + "\n")

    print(f"✅ Saved qualitative analysis to {output_path}")


if __name__ == "__main__":
    plot_training_loss()
    generate_qualitative_analysis()
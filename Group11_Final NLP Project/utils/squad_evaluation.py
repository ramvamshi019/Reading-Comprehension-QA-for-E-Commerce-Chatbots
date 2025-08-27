from sklearn.metrics import f1_score

def compute_metrics(predictions, ground_truths):
    exact_match = 0
    f1 = 0
    total = len(ground_truths)

    for qid in predictions:
        pred = predictions[qid].strip().lower()
        gt = ground_truths[qid].strip().lower()

        # Exact Match
        if pred == gt:
            exact_match += 1

        # F1 Score
        pred_tokens = pred.split()
        gt_tokens = gt.split()
        common = set(pred_tokens) & set(gt_tokens)
        if len(common) == 0:
            curr_f1 = 0
        else:
            precision = len(common) / len(pred_tokens)
            recall = len(common) / len(gt_tokens)
            curr_f1 = 2 * (precision * recall) / (precision + recall)
        f1 += curr_f1

    return {
        "exact_match": 100.0 * exact_match / total,
        "f1": 100.0 * f1 / total
    }
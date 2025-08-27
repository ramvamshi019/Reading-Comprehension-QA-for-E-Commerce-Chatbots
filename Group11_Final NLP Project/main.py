from flask import Flask, request, jsonify, render_template
import json
import difflib

app = Flask(__name__)

# Load JSON data at startup
with open('ecommerce_qa_data_ordered.json', 'r', encoding='utf-8') as f:
    ecommerce_data = json.load(f)

# Build searchable QA list
qa_data = []
for section in ecommerce_data.get("data", []):
    if "paragraphs" in section:
        for para in section["paragraphs"]:
            context = para.get("context", "").lower()
            for qa in para.get("qas", []):
                question = qa.get("question", "").lower()
                answers = qa.get("answers", [])
                answer_text = answers[0]["text"] if answers else "No answer provided"
                qa_data.append({
                    "question": question,
                    "context": context,
                    "answer": answer_text
                })
    elif all(k in section for k in ("question", "context", "answer")):
        # Handle flat QA format
        qa_data.append({
            "question": section["question"].lower(),
            "context": section["context"].lower(),
            "answer": section["answer"]
        })
    else:
        print("Warning: Unrecognized section format:", section)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    user_message = data.get('message', '').lower()

    best_match = None
    best_score = 0.0

    for qa in qa_data:
        score_q = difflib.SequenceMatcher(None, user_message, qa["question"]).ratio()
        score_c = difflib.SequenceMatcher(None, user_message, qa["context"]).ratio()
        max_score = max(score_q, score_c)

        if max_score > best_score:
            best_score = max_score
            best_match = qa["answer"]

    if best_score > 0.5:
        return jsonify({'response': best_match})
    else:
        return jsonify({'response': "I'm sorry, I couldn't understand. Could you please rephrase?"})

if __name__ == '__main__':
    app.run(debug=True)





import json
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast


def load_squad(path):   
    with open(path, 'r') as f:
        squad_dict = json.load(f)

    contexts = []
    questions = []
    answers = []
    is_impossible = []

    for group in squad_dict['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']
                impossible = qa.get('is_impossible', False)

                if impossible:
                    contexts.append(context)
                    questions.append(question)
                    answers.append({'text': '', 'answer_start': 0})
                    is_impossible.append(True)
                else:
                    for answer in qa['answers']:
                        contexts.append(context)
                        questions.append(question)
                        answers.append(answer)
                        is_impossible.append(False)

    return contexts, questions, answers, is_impossible


def add_token_positions(encodings, answers):
    start_positions = []
    end_positions = []

    for i in range(len(answers)):
        if answers[i]['text'] == '':
            start_positions.append(0)
            end_positions.append(0)
        else:
            start_char = answers[i]['answer_start']
            end_char = start_char + len(answers[i]['text'])
            offsets = encodings.offset_mapping[i]

            start_idx = end_idx = 0
            for idx, (start, end) in enumerate(offsets):
                if start <= start_char < end:
                    start_idx = idx
                if start < end_char <= end:
                    end_idx = idx
                    break

            start_positions.append(start_idx)
            end_positions.append(end_idx)

    encodings.update({
        'start_positions': torch.tensor(start_positions),
        'end_positions': torch.tensor(end_positions)
    })


class SquadDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {
            key: torch.tensor(val[idx]) 
            for key, val in self.encodings.items() 
            if key != 'offset_mapping'
        }

    def __len__(self):
        return len(self.encodings['input_ids'])


if __name__ == "__main__":
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    print("Loading data...")
    contexts, questions, answers, is_impossible = load_squad(r"C:\Users\Ramva\OneDrive\Desktop\Group11_FinalProject\Group11_Final NLP Project\data\squad_v2\train-v2.0.json")


    print("Tokenizing...")
    encodings = tokenizer(
        questions,
        contexts,
        truncation=True,
        padding=True,
        return_offsets_mapping=True
    )

    print("Aligning token positions...")
    add_token_positions(encodings, answers)

    print("Creating dataset...")
    dataset = SquadDataset(encodings)

    print(f"✅ Successfully loaded {len(dataset)} samples.")
import json
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast


def load_squad(path):
    with open(path, 'r') as f:
        squad_dict = json.load(f)

    contexts = []
    questions = []
    answers = []
    is_impossible = []

    for group in squad_dict['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']
                impossible = qa.get('is_impossible', False)

                if impossible:
                    contexts.append(context)
                    questions.append(question)
                    answers.append({'text': '', 'answer_start': 0})
                    is_impossible.append(True)
                else:
                    for answer in qa['answers']:
                        contexts.append(context)
                        questions.append(question)
                        answers.append(answer)
                        is_impossible.append(False)

    return contexts, questions, answers, is_impossible


def add_token_positions(encodings, answers):
    start_positions = []
    end_positions = []

    for i in range(len(answers)):
        if answers[i]['text'] == '':
            start_positions.append(0)
            end_positions.append(0)
        else:
            start_char = answers[i]['answer_start']
            end_char = start_char + len(answers[i]['text'])
            offsets = encodings.offset_mapping[i]

            start_idx = end_idx = 0
            for idx, (start, end) in enumerate(offsets):
                if start <= start_char < end:
                    start_idx = idx
                if start < end_char <= end:
                    end_idx = idx
                    break

            start_positions.append(start_idx)
            end_positions.append(end_idx)

    encodings.update({
        'start_positions': torch.tensor(start_positions),
        'end_positions': torch.tensor(end_positions)
    })


class SquadDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {
            key: torch.tensor(val[idx]) 
            for key, val in self.encodings.items() 
            if key != 'offset_mapping'
        }

    def __len__(self):
        return len(self.encodings['input_ids'])


if __name__ == "__main__":
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    print("Loading data...")
    contexts, questions, answers, is_impossible = load_squad("/Users/sudhachinmai/Desktop/project/Data/squad_v2/train-v2.0.json")

    print("Tokenizing...")
    encodings = tokenizer(
        questions,
        contexts,
        truncation=True,
        padding=True,
        return_offsets_mapping=True
    )

    print("Aligning token positions...")
    add_token_positions(encodings, answers)

    print("Creating dataset...")
    dataset = SquadDataset(encodings)

    print(f"✅ Successfully loaded {len(dataset)} samples.")

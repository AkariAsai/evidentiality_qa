import json
import random
import argparse
import csv
import os
from tqdm import tqdm
from collections import Counter
import string
import numpy as np
import re
import pandas as pd

def _metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


# answer nomalization
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


# F1 score definition
def _f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_fp", default=None, type=str)
    parser.add_argument("--prev_fp", default=None, type=str)
    parser.add_argument("--final_preds", default=None, type=str)
    parser.add_argument("--output_dir", default=None, type=str)
    parser.add_argument("--top_k", default=5, type=int)
    parser.add_argument("--sup_fp", default=None, type=str)

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print("loading input data")
    input_data = json.load(open(args.input_fp))
    final_preds = open(args.final_preds).read().split("\n")[:-1]
    final_preds = [pred.split("\t")[1] for pred in final_preds]
    assert len(input_data) == len(final_preds)

    ctx_dictionary = {}
    answers_dictionary = {}
    for i in range(len(input_data) // args.top_k):
        orig_question = input_data[args.top_k * i]["question"]
        ctx_first = input_data[args.top_k * i + 1]["ctxs"][0]
        ctx_rest = input_data[args.top_k * i]["ctxs"]
        ctx = [ctx_first] + ctx_rest
        assert len(ctx) == args.top_k
        assert len(set([item["question"]
                   for item in input_data[args.top_k * i: args.top_k * i + args.top_k]])) == 1
        ctx_dictionary[orig_question] = ctx
        answers_dictionary[orig_question] = input_data[args.top_k * i]["answers"]

    pos_count, neg_count = 0, 0
    no_significant_gap = 0
    positive_last = []
    negative_last = []
    # multi question and predictions
    for i in tqdm(range(len(input_data) // args.top_k)):
        orig_question = input_data[args.top_k * i]["question"]
        answer = input_data[args.top_k * i]["answers"]
        question = "{0} <answer> {1}".format(orig_question, answer[0])

        per_q_preds = final_preds[args.top_k * i: args.top_k * i + args.top_k]
        f1_scores = []
        for j, pred in enumerate(per_q_preds):
            f1_scores.append(_metric_max_over_ground_truths( _f1_score, pred, answer))

        for j, pred in enumerate(per_q_preds):
            if np.mean(f1_scores[:j] + f1_scores[j+1:]) - f1_scores[j] < -0.1:
                # without jth passage, actually the model gets more accurate results
                negative_last.append({"text": "<title> {0} <text> {1}".format(
                    ctx_dictionary[orig_question][j]["title"], ctx_dictionary[orig_question][j]["text"]), "question": question.replace("\n", ":::"), "label": "negative"})
                neg_count += 1
            elif np.mean(f1_scores[:j] + f1_scores[j+1:]) - f1_scores[j] > 0.1:
                # jth passage actually contribute to the improvements.
                positive_last.append({"text": "<title> {0} <text> {1}".format(
                    ctx_dictionary[orig_question][j]["title"], ctx_dictionary[orig_question][j]["text"]), "question": question.replace("\n", ":::"), "label": "positive"})
                pos_count += 1
            else:
                no_significant_gap += 1

    print("positive passages:{0}, negative passages{1}".format(
        pos_count, neg_count))
    print("# of examples with no significant gap {0}".format(no_significant_gap))
    print("positive: {0}".format(positive_last[-1]))
    print("negative: {0}".format(negative_last[-1]))
    final_data = positive_last + negative_last
    random.shuffle(final_data)
    # convert data format
    
    if args.sup_fp is not None:
        df = pd.read_csv(args.sup_fp)
        for index, row in df.iterrows():
            final_data.append({"text": row["sentence2"], "question": row["sentence1"], "label": row["label"]})
        random.shuffle(final_data)

    with open(os.path.join(args.output_dir, "train.csv"), "w") as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(["sentence1", "sentence2", "label"])
        for item in tqdm(final_data):
            writer.writerow([item["question"], item["text"], item["label"]])

if __name__ == "__main__":
    main()

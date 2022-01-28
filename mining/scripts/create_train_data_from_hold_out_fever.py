import json
import random
import argparse
import csv
import os
from tqdm import tqdm
from collections import Counter
import pandas as pd

def Most_Common(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_fp", default=None, type=str)
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
        orig_question = input_data[args.top_k * i ]["question"]
        ctx_first = input_data[args.top_k * i + 1]["ctxs"][0]
        ctx_rest = input_data[args.top_k * i ]["ctxs"]
        ctx = [ctx_first] + ctx_rest
        # check if the number of total passages is collect 
        assert len(ctx) == args.top_k
        # check if the passages are all for the same target questions.
        assert len(set([item["question"] for item in input_data[args.top_k * i: args.top_k * i + args.top_k ]]) )== 1
        ctx_dictionary[orig_question] = ctx
        answers_dictionary[orig_question] = input_data[args.top_k * i ]["answers"]

    pos_count, neg_count = 0, 0
    positive_last = []
    negative_last = []
    # multi question and predictions 
    for i in range(len(input_data) // args.top_k):
        orig_question = input_data[args.top_k * i ]["question"]
        answer = input_data[args.top_k * i ]["answers"][0]
        question = "{0} <answer> {1}".format(orig_question, answer)

        per_q_preds = final_preds[args.top_k * i: args.top_k * i + args.top_k ]
        for j, pred in enumerate(per_q_preds):
            other_pred_majority = Most_Common(per_q_preds[:j] + per_q_preds[j+1:])
            others_un = list(set(per_q_preds[:j] + per_q_preds[j+1:]))
            # when and only when 
            if len(others_un) == 1 and others_un[0] != pred and pred == input_data[args.top_k * i ]["answers"][0]:
                negative_last.append({"text": "<title> {0} <text> {1}".format(ctx_dictionary[orig_question][j]["title"], ctx_dictionary[orig_question][j]["text"]), "question": question, "label": "negative"})
                neg_count += 1
            elif len(others_un) == 1 and others_un[0] != pred and pred != input_data[args.top_k * i ]["answers"][0]:
                positive_last.append({"text": "<title> {0} <text> {1}".format(ctx_dictionary[orig_question][j]["title"], ctx_dictionary[orig_question][j]["text"]), "question": question, "label": "positive"})
                pos_count += 1
            elif len(others_un) == 1 and others_un[0] == pred and pred != input_data[args.top_k * i ]["answers"][0]:
                negative_last.append({"text": "<title> {0} <text> {1}".format(ctx_dictionary[orig_question][j]["title"], ctx_dictionary[orig_question][j]["text"]), "question": question, "label": "negative"})
                neg_count += 1

    print("positive passages:{0}, negative passages{1}".format(pos_count, neg_count))
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
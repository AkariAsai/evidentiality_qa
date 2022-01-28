import json
import random
import argparse
import csv
import os
from tqdm import tqdm
import jsonlines
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_fp", default=None, type=str)
    parser.add_argument("--final_preds", default=None, type=str)
    parser.add_argument("--output_dir", default=None, type=str)
    parser.add_argument("--top_k", default=20, type=int)

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
        question = input_data[args.top_k * i ]["question"]
        ctx_first = input_data[args.top_k * i + 1]["ctxs"][0]
        ctx_rest = input_data[args.top_k * i ]["ctxs"]
        ctx = [ctx_first] + ctx_rest
        # check if the number of total passages is collect 
        assert len(ctx) == args.top_k
        # check if the passages are all for the same target questions.
        assert len(set([item["question"] for item in input_data[args.top_k * i: args.top_k * i + args.top_k ]]) )== 1
        ctx_dictionary[question] = ctx
        answers_dictionary[question] = input_data[args.top_k * i ]["answers"]

    pred_dictionary = {}
    pos_count, neg_count = 0,0
    # multi question and predictions 
    for i in range(len(input_data) // args.top_k):
        question = input_data[args.top_k * i ]["question"]
        per_q_preds = final_preds[args.top_k * i: args.top_k * i + args.top_k ]
        pred_dictionary[question] = per_q_preds
    
    final_data = []
    positive_last = None
    negative_last = None
    # multi-final_answers and select positive and negative passages
    # positive passages ==> all others are collect, only fail i-th one
    # negative ==> all others are incorrect, only succeeds ith one is removed
    for question in pred_dictionary:
        preds = pred_dictionary[question]
        ctxs = ctx_dictionary[question]
        answers = answers_dictionary[question]
        pred_count = {}
        for i, pred in enumerate(preds):
            pred_count.setdefault(pred, [])
            pred_count[pred].append(i)
        print(pred_count)
        for answer in answers:
            # when and only when 19 passages are succeeded
            if answer in pred_count and len(pred_count[answer]) == args.top_k-1:
                positive_idx = [i for i in range(20) if i not in pred_count[answer]][0]
                positive_ctx = ctxs[positive_idx]
                context = "<title> {0} <text> {1}".format(positive_ctx["title"], positive_ctx["text"])
                final_data.append({"text": context, "question": "{0} <answer> {1}".format(question, answer), "label": "positive"})
                positive_last = {"text": context, "question": "{0} <answer> {1}".format(question, answer), "label": "positive"}
                pos_count += 1
                negative_samples = random.sample(pred_count[answer], k=3)
                negative_ctx = [ctxs[i] for i in negative_samples]
                for neg in negative_ctx:
                    context = "<title> {0} <text> {1}".format(neg["title"], neg["text"])
                    final_data.append({"text": context, "question": "{0} <answer> {1}".format(question, answer), "label": "negative"})
                    negative_last = {"text": context, "question": "{0} <answer> {1}".format(question, answer), "label": "negative"}
                    neg_count += 1
            elif answer in pred_count and len(pred_count[answer]) == 1:
                negative_idx = pred_count[answer][0]
                negative_ctx = ctxs[negative_idx]
                context = "<title> {0} <text> {1}".format(negative_ctx["title"], negative_ctx["text"])
                final_data.append({"text": context, "question": "{0} <answer> {1}".format(question, answer), "label": "negative"})
                negative_last = {"text": context, "question": "{0} <answer> {1}".format(question, answer), "label": "negative"}
    
    print("positive passages:{0}, negative passages{1}".format(pos_count, neg_count))
    print("positive: {0}".format(positive_last))
    print("negative: {0}".format(negative_last))

    if args.sup_fp is not None:
        df = pd.read_csv(args.sup_fp)
        for index, row in df.iterrows():
            final_data.append({"text": row["sentence2"], "question": row["sentence1"], "label": row["label"]})
        random.shuffle(final_data)
    # convert data format
    with open(os.path.join(args.output_dir, "train.csv"), "w", newline='\n') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(["sentence1", "sentence2", "label"])
        for item in tqdm(final_data):
            writer.writerow([item["question"], item["text"], item["label"]])

if __name__ == "__main__":
    main()
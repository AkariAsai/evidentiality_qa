import json
import random
import argparse
import csv
import os
from tqdm import tqdm
import jsonlines


def convert_data_classification(data, positive_ratio=1.0, negative_ratio=0.1, top_k=50, split="train"):
    data_positives = []
    data_negatives = []
    for q_id, item in enumerate(data):
        question = item["question"]
        answers = item["answers"]
        ctxs = item["ctxs"][:top_k]
        for ctx in ctxs:
            context = "<title> {0} <text> {1}".format(
                ctx["title"], ctx["text"])
            answer_found = False
            for answer in answers:
                if answer in ctx["text"]:
                    continue
                    # data_positives.append(
                    #     {"text": context, "question": question, "label": "positive"})
                    # answer_found = True
            if answer_found is False:
                data_negatives.append(
                    {"text": context, "question": "{0} <answer> {1}".format(question, answers[0]), "label": "negative"})
    if split == "train":
        # data_positives = random.sample(data_positives, k=int(
        #     len(data_positives) * positive_ratio))
        data_negatives = random.sample(data_negatives, k=int(
            len(data_negatives) * negative_ratio))
        # print("# pos: {0}, # neg: {1}".format(
        #     len(data_positives), len(data_negatives)))

    # final_data = data_positives + data_negatives
    final_data = data_negatives
    random.shuffle(final_data)

    return final_data


def convert_data_classification_test(data, top_k=50):
    final_data = []
    for q_id, item in enumerate(data):
        question = item["question"]
        ctxs = item["ctxs"][:top_k]
        answers = item["answers"]
        for ctx in ctxs:
            context = "<title> {0} <text> {1}".format(
                ctx["title"], ctx["text"])
            label = "negative"
            for answer in answers:
                if answer in ctx["text"]:
                    label = "positive"

            final_data.append(
                {"text": context, "question": "{0} <answer> {1}".format(question, answers[0]), "label": label})

    return final_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_fp", default=None, type=str)
    parser.add_argument("--dev_fp", default=None, type=str)
    parser.add_argument("--test_fp", default=None, type=str)
    parser.add_argument("--pos_ratio", default=1.0, type=float)
    parser.add_argument("--neg_ratio", default=0.1, type=float)
    parser.add_argument("--top_k", default=50, type=int)
    parser.add_argument("--output_dir", default=None, type=str)

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.train_fp:
        train_data = json.load(open(args.train_fp))
        final_train_data = convert_data_classification(
            train_data, args.pos_ratio, args.neg_ratio, args.top_k)

        header = ['sentence1', 'sentence2', 'label']
        with open(os.path.join(args.output_dir, "train.csv"), "w", newline='') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(header)
            for item in tqdm(final_train_data):
                writer.writerow([item["question"], item["text"], item["label"]])

    if args.dev_fp:
        dev_data = json.load(open(args.dev_fp))
        final_dev_data = convert_data_classification(
            dev_data, args.pos_ratio, args.neg_ratio, args.top_k)

        header = ['sentence1', 'sentence2', 'label']
        with open(os.path.join(args.output_dir, "dev.csv"), "w", newline='') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(header)
            for item in tqdm(final_dev_data):
                writer.writerow([item["question"], item["text"], item["label"]])


    if args.test_fp:
        test_data = json.load(open(args.test_fp))
        final_dev_data = convert_data_classification_test(test_data, top_k=args.top_k)

        header = ['sentence1', 'sentence2', 'label']
        with open(os.path.join(args.output_dir, "test.csv"), "w", newline='') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(header)
            for item in tqdm(final_dev_data):
                writer.writerow([item["question"], item["text"], item["label"]])
if __name__ == "__main__":
    main()

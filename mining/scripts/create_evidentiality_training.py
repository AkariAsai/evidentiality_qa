import json
import random
import argparse
import csv
import os
from tqdm import tqdm
import jsonlines
import copy
import random
import numpy as np

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

def assign_positive_negative(input_data, score_dict, topn=100, para_limit=20, shuffle=False):
    new_data = []
    shuffled = 0
    recall_orig = 0
    recall_shuffle = 0
    for q_idx, item in tqdm(enumerate(input_data)):
        new_item = copy.deepcopy(item)
        for p_idx, ctx in enumerate(new_item["ctxs"]):
            global_p_idx = q_idx * topn + p_idx

            pred_score = softmax(score_dict[global_p_idx]["score"])[1]
            if pred_score > 0.35:
                pred = "positive"
            else:
                pred = "negative"
            # pred = score_dict[global_p_idx]["pred"]
            if pred == "negative":
                ctx["has_answer"] = False
            else:
                ctx["has_answer"] = True
        
        if len([ctx for ctx in new_item["ctxs"][:para_limit] if ctx["has_answer"] is True]) > 0:
            recall_orig += 1

        if shuffle is True and len([ctx for ctx in new_item["ctxs"][:para_limit] if ctx["has_answer"] is True]) == 0 and len([ctx for ctx in new_item["ctxs"][para_limit:] if ctx["has_answer"] is True]) > 0:
            for p_idx, ctx in enumerate(new_item["ctxs"][para_limit:]):
                if ctx["has_answer"] is True:
                    random_idx = random.sample(range(0, 5), k=1)[0]
                    prev = new_item["ctxs"][random_idx]
                    new_item["ctxs"][random_idx] = ctx
                    new_item["ctxs"][p_idx + para_limit] = prev
                    shuffled += 1
                    break

        if len([ctx for ctx in new_item["ctxs"][:para_limit] if ctx["has_answer"] is True]) > 0:
            recall_shuffle += 1

        new_data.append(new_item)
    
    print("final recall:")
    print("original recall @ {0}: {1}".format(recall_orig / len(input_data), para_limit))
    print("shuffle recall @ {0}: {1}".format(recall_shuffle / len(input_data), para_limit))
    return new_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_fp", default=None, type=str)
    parser.add_argument("--score_fp", default=None, type=str)
    parser.add_argument("--top_k", default=100, type=int)
    parser.add_argument("--para_limit", default=20, type=int)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--output_dir", default=None, type=str)

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    input_data = json.load(open(args.input_fp))
    score_data = json.load(open(args.score_fp))
    score_data = {int(id): data for id, data in score_data.items()}

    final_sorted_data = assign_positive_negative(input_data, score_data, topn=args.top_k, para_limit=args.para_limit, shuffle=args.shuffle)

    assert input_data != final_sorted_data
    with open(os.path.join(args.output_dir, "sorted_test.json"), "w") as outfile:
        json.dump(final_sorted_data, outfile)

if __name__ == "__main__":
    main()

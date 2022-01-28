import json
import argparse
from tqdm import tqdm
import os
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_fp",
                        default=None, type=str)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--max", type=int, default=50)
    parser.add_argument("--output_fp", default=None, type=str)
    parser.add_argument("--more", action="store_true")
    args = parser.parse_args()

    input_data = json.load(open(args.input_fp))
    new_data = []

    for item in input_data:
        if args.more is True:
            for i in range(len(item["ctxs"][:args.max]) // args.top_k):
                ctxs = item["ctxs"][i * args.top_k: (i+1) * args.top_k]
                for passage_idx in range(args.top_k):
                    new_ctx = ctxs[:passage_idx] + ctxs[passage_idx + 1 :]
                    assert len(new_ctx) == args.top_k - 1
                    question = item["question"]
                    answers = item["answers"]
                    new_data.append({"question": question, "answers": answers, "ctxs": new_ctx})
        else:
            ctxs = item["ctxs"][:args.top_k]
            for passage_idx in range(args.top_k):
                new_ctx = ctxs[:passage_idx] + ctxs[passage_idx + 1 :]
                assert len(new_ctx) == args.top_k - 1
                question = item["question"]
                answers = item["answers"]
                new_data.append({"question": question, "answers": answers, "ctxs": new_ctx})
            
    print("{} data created:".format(len(new_data)))

    with open(args.output_fp, "w") as outfile:
        json.dump(new_data, outfile)

if __name__ == "__main__":
    main()
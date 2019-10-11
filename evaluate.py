import json
import pickle
from metrics.evaluators import f_map, f_recall
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("gold")
    parser.add_argument("results")
    parser.add_argument("outfile")

    args = parser.parse_args()

    with open(args.gold, "r") as f:
        gs = json.load(f)
        gs = {x["query_id"]: {"documents": x["documents"], "query": x["query"]} for x in gs}

    with open(args.results, "rb") as f:
        r = pickle.load(f)

    predictions = []
    expectations = []

    for _id in gs.keys():
        expectations.append(gs[_id]["documents"])
        predictions.append(list(map(lambda x: x["id"], r[_id]["documents"])))

    bioasq_map = f_map(predictions, expectations, bioASQ=True)
    str_bioasq_map = "[DEEPRANK] BioASQ MAP@10: {}".format(bioasq_map)
    print(str_bioasq_map)
    str_map = "[DEEPRANK] Normal MAP@10: {}".format(f_map(predictions, expectations))
    print(str_map)
    str_recall = "[DEEPRANK] Normal RECALL@{}: {}".format(10, f_recall(predictions, expectations, at=10))
    print(str_recall)

    # get false positives
    fp = []

    for _id in gs.keys():
        expectation = gs[_id]["documents"]
        docs_fp = []

        for i, doc in enumerate(r[_id]["documents"]):
            if doc["id"] not in expectation:
                docs_fp.append({"pmid": doc["id"], "text": doc["original"], "rank_position": i})

        fp.append({"query_id": _id, "query": gs[_id]["query"], "recall": (10-len(docs_fp))/min(len(gs[_id]["documents"]), 10), "documents": docs_fp})

    with open(args.outfile, "w") as f:
        json.dump(fp, f)

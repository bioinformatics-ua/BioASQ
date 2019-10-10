"""
Application entry point
"""

import argparse
from pipeline import Pipeline
from logger import log
import os
import pickle


if __name__ == "__main__":
    log.info("[Execute]")
    # ARG parse
    parser = argparse.ArgumentParser(description="Neural Information Retrieval pipeline")
    parser.add_argument("--mode", help="Runing mode of the pipeline",
                        choices=['train', 'inference'])
    parser.add_argument("--query", help="Runing mode of the pipeline")
    parser.add_argument("--queries", help="Validation file with queries")
    parser.add_argument("config", help="configuration file with the instructions, must be in json or yaml format")

    args = parser.parse_args()

    if args.query is not None:
        print("[MODE]: Inference for query", args.query)
        mode = "inference"
    if args.queries is not None:
        print("[MODE]: Test for file", args.queries)
        mode = "test"
    else:
        print("[MODE]: Train")
        mode = "train"
    # create Pipeline Object
    print()
    pipeline = Pipeline(args.config, mode)
    pipeline.build()

    if mode == "train":
        print("---------------------\n[ROUTINE] Steps that the pipeline for TRAIN will execute")
        steps = pipeline.train(simulation=True)["steps"]
        for step in steps:
            print("\t", step)
        print("---------------------")
        pipeline.train()
    elif mode == "inference":
        print("---------------------\n[ROUTINE] Steps that the pipeline for INFERENCE will execute")
        steps = pipeline.inference(simulation=True, query=args.query)["steps"]
        for step in steps:
            print("\t", step)
        print("---------------------")
        pipeline.inference(query=args.query)
    elif mode == "test":
        print("---------------------\n[ROUTINE] Steps that the pipeline for INFERENCE will execute")
        steps = pipeline.inference(simulation=True, queries_file=args.queries)["steps"]
        for step in steps:
            print("\t", step)
        print("---------------------")
        name = "results_"+os.path.basename(args.queries)+".p"
        abspath = os.path.abspath(os.path.basename(args.queries))
        print(os.path.join(abspath, name))

        #retrieved = pipeline.inference(queries_file=args.queries)["retrieved"]
        #print("Save results")

        #with open(os.path.join(abspath, name), "wb") as f:
        #    pickle.dump(retrieved, f)

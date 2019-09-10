"""
Application entry point
"""

import argparse
from pipeline import Pipeline
from logger import log


if __name__ == "__main__":
    log.info("[Execute]")
    # ARG parse
    parser = argparse.ArgumentParser(description="Neural Information Retrieval pipeline")
    parser.add_argument("--mode", help="Runing mode of the pipeline",
                        choices=['train', 'inference'])
    parser.add_argument("config", help="configuration file with the instructions, must be in json or yaml format")

    args = parser.parse_args()

    # create Pipeline Object
    print()
    pipeline = Pipeline(args.config, args.mode)
    pipeline.build()

    print("---------------------\n[ROUTINE] Steps that the pipeline will execute")
    steps = pipeline.train(simulation=True)["steps"]
    for step in steps:
        print("\t", step)
    print("---------------------")

    pipeline.train()

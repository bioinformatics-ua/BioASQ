"""
Application entry point
"""

import argparse
from pipeline import Pipeline
import logging

# Setup logging
logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')


if __name__ == "__main__":

    # ARG parse
    parser = argparse.ArgumentParser(description="Neural Information Retrieval pipeline")
    parser.add_argument("--mode", help="Runing mode of the pipeline",
                        choices=['train', 'inference'])
    parser.add_argument("config", help="configuration file with the instructions, must be in json or yaml format")

    args = parser.parse_args()

    # create Pipeline Object
    print()
    pipeline = Pipeline(args.config, args.mode, logging)
    pipeline.build()

    pipeline.check_train_routine()

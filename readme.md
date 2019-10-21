# Generic pipeline for Information Retrieval and BioDeepRank
NOTE: The information realted to BioDeepRank will be added during the next days...
The notebook Interaction


## Description
This repository implements a generic pipeline that can be used to address the majoraty of the IR tasks.

The pipeline is described by a configuration file in yaml, which facilitate the prototyping and testing of complex IR models.

The motivation is to offer an infrastructure that allows multiple tests (fine-tunning) of complex models in multiples datasets.


## Pipeline building blocks

The first input to the pipeline is allways the dataset corpora and the queries (if appliable). This input is fed to a chain of modules, where the output of the previous one is fed to as input of the next one.

Each module is dynamicly loaded in runtime, which gives a high degree of freedom, since each module can be fully customized and added to the pipeline.

![Image of Yaktocat](images/pipe.png)

## Pipeline configuration file

folder config have multiple examples

(TODO)

```yaml
cache_folder: "/path/to/folder"
corpora:
    name: "bioasq"
    folder: "path/to/folder.tar.gz" #corresponds to the tar.gz file
    files_are_compressed: true #(optinal) default is false
queries:
    folder: "path/to/folder"
pipeline:
    - BM25:
        top_k: 2500
        tokenizer:
            Regex:
                stem: true
    - DeepRank:
        top_k: 10
        input_network:
            Q: 13 #number max of query tokens
            P: 5 #number max of snippets per query token
            S: 15 #number max of snippet tokens
            embedding_matrix: "auto" #creates a embedding matrix using fasttext library
        measure_network: "MeasureNetwork" #class name of the measure network
        aggregation_network: "AggregationNetwork" #class name of the aggregation network
        hyperparameters:
            optimizer: "adadelta" #(optinal) default is AdaDelta
            l2_regularization: 0.0001 #(optinal) default is 0.0001
            num_partially_positive_samples: 3
            num_negative_samples: 4
        tokenizer:
            Regex:
                stem: false

```

## Run
Train
```sh
python3 main.py config_example.yaml

```

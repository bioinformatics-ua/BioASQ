import os
import json
from collections import Counter
BioASQ_training_file = "BioASQ-trainingDataset6b.json"

#resolve paths
if os.path.basename(os.getcwd()) == "preprocess":
    path = os.path.join("..", "data", BioASQ_training_file)
elif os.path.basename(os.getcwd()) == "bioASQ-taskb":
    path = os.path.join(".", "data", BioASQ_training_file)
else:
    raise RuntimeError("current folder "+os.path.basename(os.getcwd())+" is invalid")

print("Current data path:",path)

data = json.load(open(path))["questions"]

print("Number of questions: ",len(data))

#count questions by type
print(Counter([ q["type"] for q in data ]))

#visualize factoid exact and ideal anwsers
factoid_exact_ideal = [ (q["body"],q["exact_answer"],q["ideal_answer"]) for q in data if q["type"]=="factoid"]


#find articles that dont have snippets



[ [ (q["type"],q["exact_answer"],q["ideal_answer"]) for q in data if q["type"]=="factoid"]]


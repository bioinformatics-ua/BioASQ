import os

def resolve_path(path):
    if os.path.basename(os.getcwd()) == "preprocess":
        return os.path.join("..", "data", path)
    elif os.path.basename(os.getcwd()) == "bioASQ-taskb":
        return os.path.join(".", "data", path)
    else:
        raise RuntimeError("current folder " + os.path.basename(os.getcwd()) + " is invalid")

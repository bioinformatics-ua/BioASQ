import sys

def mem_vars_notebook():
    """
    This function will print defined variables on the notebook
    
    Variables that start with _ will not appear
    """
    a = 1

    ipython_vars = []# ['In', 'Out', 'exit', 'quit', 'get_ipython', 'ipython_vars']
    
    print(sorted([(x, sys.getsizeof(globals().get(x))) for x in dir() if not x.startswith('_') and x not in sys.modules and x not in ipython_vars], key=lambda x: x[1], reverse=True))

def f_recall(predictions, expectations, at=2500):
    """
    predictions: list of list of pmid
    expectations: list of list of pmid
    """

    assert len(predictions) == len(expectations)

    return sum([__recall(predictions[i][:at], expectations[i]) for i in range(len(predictions))])/len(predictions)


def __recall(prediction, expectation):
    """
    prediction: list of cut at-k pmid each element should be a tuple (pmid,score) or (pmid)
    expectation: list of valid pmid

    return (precision, binary relevance list)
    """
    #normalization
    if isinstance(prediction[0], tuple):
        prediction = list(map(lambda x: [0], prediction))

    return sum([1 if pmid in expectation else 0 for pmid in prediction])/len(expectation)


def __precision(prediction, expectation):
    """
    prediction: list cut at-k pmid, each element should be a tuple (pmid,score) or (pmid)
    expectation: list of valid pmid

    return precision
    """
    #normalization
    if isinstance(prediction[0], tuple):
        prediction = list(map(lambda x:x[0], prediction))

    return sum([ 1 if pmid in expectation else 0 for pmid in prediction])/len(prediction)


def __average_precision_at(prediction, expectation, bioASQ, use_len=False, at=10):
    """
    predictions: list of pmid, each element can be a tuple (pmid,score) or (pmid)
    expectations: list of valid pmid

    return average precision at k
    """

    assert len(prediction)>0

    # normalization
    if isinstance(prediction[0], tuple):
        prediction = list(map(lambda x:x[0], prediction))

    binary_relevance = [ 1 if pmid in expectation else 0 for pmid in prediction[:at] ]
    precision_at = [ __precision(prediction[:i],expectation) for i in range(1,at+1) ]

    if bioASQ:
        return sum([a*b for a,b in zip(precision_at,binary_relevance)])/10
    elif use_len:
        return sum([a*b for a,b in zip(precision_at,binary_relevance)])/len(expectation)
    elif sum(binary_relevance)>0:
        return sum([a*b for a,b in zip(precision_at,binary_relevance)])/sum(binary_relevance)
    else: #The indetermination 0/0 will be consider 0
        return 0

def f_map(predictions, expectations, at=10,bioASQ = False,use_len=False):
    """
    predictions: list of list of pmid
    expectations: list of list of pmid
    """
    assert len(predictions) == len(expectations)

    return sum([ __average_precision_at(predictions[j],expectations[j],bioASQ,use_len, at) for j in range(len(predictions))])/len(predictions)
    #for query_predictions in range(len(predictions)):

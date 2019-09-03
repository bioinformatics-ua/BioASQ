from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS, cross_origin

import json
from os.path import join
import os
import sys
from tensorflow.keras.preprocessing.sequence import pad_sequences
import time

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

module_path = os.path.abspath(os.path.join('../pubmed_data'))
if module_path not in sys.path:
    sys.path.append(module_path)

from models.fast_neural_retrieval.elasticsearch_bm25_model import BM25_ES
from models.deep_model_for_ir.deeprank_v6_model import DeepRankV6

from pubmed_data import pubmed_helper as ph
import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
#deep_model = DeepRankV6.load()

def init():
    global fast_model, deep_model, get_snippet_attn, get_term_attn, graph
    
    fast_model = BM25_ES()
    deep_model = DeepRankV6.load()
    snippet_attention_tensor = deep_model.document_score_model.layers[11].attention_weights
    q_term_attention_tensor = deep_model.document_score_model.layers[14].attention_weights

    get_snippet_attn = K.function(deep_model.document_score_model.input, snippet_attention_tensor)
    get_term_attn = K.function(deep_model.document_score_model.input, [q_term_attention_tensor])

    graph = tf.get_default_graph()
    
    
app = Flask(__name__,static_folder='./build/')
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'



# Serve React App
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    print(path)
    print(app.static_folder + path)
    print(os.path.exists(app.static_folder + path))
    if path != "" and os.path.exists(app.static_folder + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

@app.route('/api', methods=['POST'])
@cross_origin()
def run_system():
    content = request.get_json()
    
    query = {"body":content["query"], "id":"_"}
    
    bm25_time = time.time()
    fast_result = fast_model.predict([query],top_k=2500)["_"]#drop query_id
    #{body:<query>, documents:[(pmid,score,text,title)]}
    print("BM 25 Time:",time.time()-bm25_time)
    
    #Data preparation
    X = _prepare_single_query(deep_model, fast_result)
    
    
    print("Start deep model")
    dr_time = time.time()
    with graph.as_default():
        deep_r = _deep_model_single_predict(deep_model, X, fast_result["documents"])[:10]
        
    print("DR Time:",time.time()-dr_time)
    #print(deep_r)
    print("Start highlight")
    hightlight_time = time.time()
    highlight_r = []
    
    for document in deep_r:
        document_index = document[1]
        document_text = fast_result["documents"][document_index][2]

        prepared_X = [np.array([X[0][document_index,:]]), np.array([X[1][document_index,:]]), np.array([X[2][document_index,:]]) ]
  
        
        query_list_string, doc_list_string = highlight_snippets(content["query"], 
                                                                document_text,
                                                                prepared_X, 
                                                                TOP_QUERY=13, 
                                                                TOP_SNIPPETS=10)
        
        highlight_r.append({"query":" ".join(query_list_string),
                            "document":"".join(doc_list_string),
                            "pmid":document[0],
                            "score":document[2],
                            "title":fast_result["documents"][document_index][3]})
    
    print("Highlight Time:",time.time()-hightlight_time)
    
    return jsonify(highlight_r)


##AUXILIAR FUNCTION FOR NEURAL MODEL
#Number max of term per query
MAX_Q_TERM = 13

#Number max of the snippet terms
QUERY_CENTRIC_CONTEX = 15

#Number max of passages per query term
MAX_PASSAGES_PER_QUERY = 5

#Snippet position padding value
SNIPPET_POSITION_PADDING_VALUE = -1
def _prepare_single_query(self, bm25_query):
    """
    {body:<query>, documents:[(pmid,score,original text),...,]}
    """
    doc_time_tokenize = []
    query = []
    query_doc = []
    query_doc_position = []
    tokenized_query = self.tokenizer.texts_to_sequences([bm25_query["body"]])[0]
    #manualy remove the stopwords
    tokenized_query = [ token for token in tokenized_query if token not in self.biomedical_stop_words_tokens]

    tokenized_query = pad_sequences([tokenized_query], maxlen = MAX_Q_TERM, padding="post")[0]

    for doc_data in bm25_query["documents"]:
        #positive
        doc_start = time.time()
        tokenized_doc = self.tokenizer.texts_to_sequences([doc_data[2]])[0]
        doc_time_tokenize.append(time.time()-doc_start)
        doc_snippets, doc_snippets_position = self.snippet_interaction(tokenized_query, tokenized_doc)
        
        ### add ###
        query.append(tokenized_query)

        #positive doc
        query_doc.append(doc_snippets)
        query_doc_position.append(doc_snippets_position)

    #missing fill the gap for the missing query_terms
    
    print("Tokenization took",sum(doc_time_tokenize),"seconds")
    return [np.array(query), np.array(query_doc), np.array(query_doc_position)]

def _deep_model_single_predict(self, X, bm25_documents):
    
    start_q_time = time.time()
    deep_ranking = self.document_score_model.predict(X)
    print("DeepRank predict time",time.time()-start_q_time)

    start_sort_time = time.time()

    index = list(range(deep_ranking.shape[0]))
    deep_ranking_pmid = list(zip(map(lambda x:x[0],bm25_documents),index,map(lambda x:x[0],deep_ranking.tolist())))
    
    
    deep_ranking_pmid.sort(key=lambda x:-x[2])
    print("DeepRank sort time",time.time()-start_sort_time)
    
    return deep_ranking_pmid

##AUXILIAR FUNCTION FOR THE HIGHLIGHT


def red_percentage_print(s, percentage):
    rescale = 100-int(percentage*100)
    return "<text style=background-color:hsl(0,100%,{}%);>{}</text>".format(rescale, s)

def blue_percentage_print(s, percentage):
    rescale = 100-int(percentage*100)
    return "<text style=background-color:hsl(220,100%,{}%);>{}</text>".format(rescale, s)

def highlight_snippets(query, document_text, X, TOP_SNIPPETS = 5, TOP_QUERY = 5):
    #X = deep_model._prepare_data(query, document_index, articles)
    
    # Query highlight
    snippet_attn = np.squeeze(np.array(get_snippet_attn(X)))
    term_attn = np.expand_dims(np.squeeze(np.array(get_term_attn(X))), axis=-1)
    
    comb_W = (snippet_attn*term_attn)
    comb_W_1d =comb_W.ravel()
    #print(comb_W_1d)
    #top 5 snippet index
    snippets_indexs = comb_W_1d.argsort()[-TOP_SNIPPETS:][::-1]
    
    #snippets_ravel_attention_normalize = comb_W_1d[snippets_indexs]/sum(comb_W_1d[snippets_indexs])
    snippets_ravel_attention =  comb_W_1d[snippets_indexs]/sum(comb_W_1d[snippets_indexs])
    #snippets_ravel_attention = snippets_ravel_attention/sum(snippets_ravel_attention)

    #print("Snippet attention normalize", snippets_ravel_attention)
    
    query_list_string = [deep_model.tokenizer.index_word[x] for x in X[0][0] if x != 0]
    
    term_attn_ravel = term_attn.ravel()
    TOP_QUERY = min(len(query_list_string),TOP_QUERY)
    top_5_q_terms = term_attn_ravel.argsort()[-TOP_QUERY:][::-1]
    highlight_terms = term_attn_ravel[top_5_q_terms]/sum(term_attn_ravel[top_5_q_terms])

    

    for count,index in enumerate(top_5_q_terms):
        if index>=len(query_list_string):
            continue
            
        query_list_string[index] = red_percentage_print(query_list_string[index],highlight_terms[count])

    # DOCUMENT highlight
    doc_tokens = deep_model.tokenizer.texts_to_sequences([document_text])[0]

    doc_list_string = [deep_model.tokenizer.index_word[x]+" " for x in doc_tokens if x != 0]

    snippet_position_ravel = X[2][0].ravel()

    for count,index in enumerate(snippets_indexs):

        index = snippet_position_ravel[index]
        if index==-1:
            continue
        low_index = max(0,index-7)
        high_index = max(0,index+7)

        doc_list_string[low_index:high_index] = list(map(lambda x:blue_percentage_print(x,snippets_ravel_attention[count]),doc_list_string[low_index:high_index]))


    return query_list_string, doc_list_string


if __name__ == '__main__':
    init()
    app.debug = True
    app.run(port=3306, host="0.0.0.0", use_reloader=False)
    
    

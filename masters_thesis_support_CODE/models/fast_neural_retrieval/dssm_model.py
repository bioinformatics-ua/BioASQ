import os 
import sys
import types
import pickle
import gc
import numpy as np
from scipy import sparse

#KERAS
from tensorflow.keras.layers import Input, Dense, Dot, Activation, Concatenate
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, LambdaCallback
import tensorflow.keras.backend as K


from models.generic_model import ModelAPI, _sim_cos, _post_predict

from pubmed_data import pubmed_helper as ph

from pubmed_data.keras_new_text import regex_alfanum_tokenizer

##add keras to the modules
module_path = os.path.abspath(os.path.join('pubmed_data'))
if module_path not in sys.path:
    sys.path.append(module_path)

ht_tokenizer = ph.load_tokenizer(mode="hashtrick_full_tokens")
#same memory
del ht_tokenizer.index_word
del ht_tokenizer.index_docs
del ht_tokenizer.word_counts
del ht_tokenizer.word_docs

TRIGRAM_VOC = len(ht_tokenizer.word_index) + 1 

def bag_of_trigram( texts):

    _matrix = np.zeros((len(texts),TRIGRAM_VOC), dtype=np.int8)

    for i,text in enumerate(texts):
        bag_of_word = regex_alfanum_tokenizer(text)
        for j in ht_tokenizer.texts_to_sequences(bag_of_word):
            _matrix[i][j] += 1

    return _matrix

def dssm_projectiom_model(activation='tanh'):
    
    dense_1 = Dense(300, 
                        activation=activation,
                        kernel_initializer='glorot_uniform', 
                        bias_initializer='glorot_uniform')
        
    dense_2 = Dense(300, 
                    activation=activation,
                    kernel_initializer='glorot_uniform', 
                    bias_initializer='glorot_uniform')

    dense_3 = Dense(128, 
                    activation=activation,
                    kernel_initializer='glorot_uniform', 
                    bias_initializer='glorot_uniform')

    def build_model(inputs, model_name):
        x = dense_1(inputs)
        x = dense_2(x)
        x = dense_3(x)
        
        return Model(inputs=[inputs], outputs=[x],name=model_name)
    
    return build_model
    
class DSSM(ModelAPI):

    def __init__(self, vocabulary_size, num_neg_examples=4, same_q_d_model = True, only_title=False ,show_model_arch=False, build=True):
        super().__init__()
        
        self.TRIGRAM_SIZE = vocabulary_size
        self.num_neg_examples = num_neg_examples
        self.show_model_arch = show_model_arch
        
        self.only_title=only_title
        if self.only_title:
            self.transform_document = lambda x:x["title"]
        else:
            self.transform_document = lambda x:x["title"]+ " " +x["abstract"]
        
        self.collection_representation = []
        self.same_q_d_model = same_q_d_model
        
            
        #Last line to be exectuted
        if build:
            self.__build_model()
        
        

    def __build_model(self):
        #Build the keras dssm model
        K.clear_session()
        

        # Follow the paper arch
        
        #The INPUT will be the result of the hash trick layer
        query = Input(shape = (self.TRIGRAM_SIZE,), name = "dssm_query_input")
        pos_doc = Input(shape = (self.TRIGRAM_SIZE,), name = "dssm_pos_doc_input")
        neg_docs = [Input(shape = (self.TRIGRAM_SIZE,), name = ("dssm_neg_doc_input_"+str(i))) for i in range(self.num_neg_examples)]
        
        
        
        #Create a sub model of the network (siamese arch)
        #2 Inputs query and doc
        q_input = Input(shape = (self.TRIGRAM_SIZE,), name= "q_input")
        doc_input = Input(shape = (self.TRIGRAM_SIZE,), name= "doc_input")
            
        if self.same_q_d_model:
            projection_input = Input(shape = (self.TRIGRAM_SIZE,), name= "projection_input")
            #same weights
            dssm_model_builder = dssm_projectiom_model()
            projection_model = dssm_model_builder(projection_input,"projection_model")

            #same model for both
            self.query_projection = projection_model
            self.doc_projection = projection_model
        else:
            
            #different weights
            self.query_projection = dssm_projectiom_model()(q_input,"query_projection")
            self.doc_projection = dssm_projectiom_model()(doc_input,"doc_projection")
        
        query_projection = self.query_projection(q_input)
        doc_projection = self.doc_projection(doc_input)
        #similarity between the query and the docs
        q_doc_sim = Dot(axes=1,normalize=True)([query_projection,doc_projection])
        
        sub_model = Model(inputs=[q_input, doc_input], outputs=[q_doc_sim], name="siamese_model")
        
        if self.show_model_arch:
            print("Sub model arch")
            sub_model.summary()
        
        #Making the softmax approximation for 1 pos doc and N neg doc
        q_doc_pos_output = sub_model([query,pos_doc])
        q_doc_neg_output = [sub_model([query,neg_doc]) for neg_doc in neg_docs]
        
        concat = Concatenate(axis=1)([q_doc_pos_output]+q_doc_neg_output)
        
        #missing the smoth factor
        prob = Activation("softmax")(concat)
        
        self.dssm_model = Model(inputs=[query,pos_doc]+neg_docs,outputs=prob)
        if self.show_model_arch:
            self.dssm_model.summary()
        
        #try the sgd optimizer
        self.dssm_model.compile(optimizer='sgd',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
       
    def _normalize_document_collection(self):
        self.collection_representation_norm = np.linalg.norm(self.collection_representation, axis=1).reshape(-1,1) #matrix dimensions [DOC,1]
    
    
    
    def create_data_generator(self, data, articles, batch, only_title=False):
        """
        Create a python generator to fed the data in batch to the model
        
        data: list of queries with the following struct {body:"question body",title:"doc title",abstract:"doc abstract"}
        articles: document collection
        """
        
        def bag_of_trigram_list_of_list(g_texts):

            return [bag_of_trigram(texts) for texts in g_texts]
    
        def negative_random_index(low, high, selection, exclude):

            #bad approach! but the selection exclude is a lot small that the num articles...
            neg_random_indexs = np.random.randint(0,len(articles),(selection,))

            while any([i in exclude for i in neg_random_indexs]):
                neg_random_indexs = np.random.randint(0,len(articles),(selection,))

            return neg_random_indexs
        
        #VER ISTO!
        with open("/backup/saved_models/pmid_index_mapping.p","rb") as f:
            pmid_document_map = pickle.load(f)        
        
        def training_generator(data, batch=batch, neg_examples=self.num_neg_examples, only_title=only_title):

            BATCH = batch #approx number of queries to return per batch
            
            q_pos_neg_doc = []

            max_article_index = len(articles)

            while True:

                for query_data in data:

                    if len(q_pos_neg_doc)>=BATCH:
                        b_tri = np.array(bag_of_trigram_list_of_list(q_pos_neg_doc))

                        q = b_tri[:,0,:]
                        pos_doc = b_tri[:,1,:]
                        neg_doc = [b_tri[:,i,:] for i in range(2,2+neg_examples)]
                        X = [q,pos_doc]+neg_doc

                        Y = np.array([[1]+[0]*neg_examples]*len(q_pos_neg_doc))

                        yield (X,Y)
                        q_pos_neg_doc = []
                    else:
                        pos_doc_set = {pmid_document_map[document_pmid] for document_pmid in query_data["documents"]}


                        for index_article in pos_doc_set:
                            row=[]
                            row.append(query_data["body"])
                            row.append(self.transform_document(articles[index_article]))

                            neg_random_indexs = negative_random_index(0, max_article_index, neg_examples, pos_doc_set)
                            row.extend([ self.transform_document(articles[neg_index]) for neg_index in neg_random_indexs])
                            q_pos_neg_doc.append(row) 
        
        return training_generator(data)
        
    def _training_process(self, data, **kwargs):
        #assume that the data is alredy in the format: (query,pos_doc,[neg_docs])
        
        if 'training_data' not in kwargs or 'validation_data' not in kwargs:
            raise TypeError('training_data and validation_data must be suplied!')
        
        training_data = kwargs.pop('training_data') 
        validation_data = kwargs.pop('validation_data') 
        
        if 'batch' in kwargs:
            batch = kwargs.pop('batch')
        else:
            batch = 1024
        
        if 'epoach' in kwargs:
            epoach = kwargs.pop('epoach')
        else:
            epoach = 20
            
        if 'only_title' in kwargs:
            only_title = kwargs.pop('only_title')
        else:
            only_title = False
        
        if 'neg_examples' in kwargs:
            self.num_neg_examples = kwargs.pop('neg_examples')
        
        if kwargs:
            raise TypeError('Unrecognized keyword arguments: ' + str(kwargs))
        
        
        
        training_samples = sum([ len(q["documents"]) for q in training_data])
        train_steps = training_samples//batch
        print("Train_steps:",train_steps)

        test_samples = sum([ len(q["documents"]) for q in validation_data])
        test_steps = test_samples//batch
        print("Test_steps:",test_steps)

        #data generators
        train_generator = self.create_data_generator(training_data,
                                                     data,
                                                     batch = batch,
                                                     only_title = only_title)
        
        validation_generator = self.create_data_generator(validation_data,
                                                     data,
                                                     batch = batch,
                                                     only_title = only_title)
        
        
        #callback
        save_best_file_name = "best_checkpoint_dssm_model_"+("title" if only_title else "") + ".h5"
        callback = ModelCheckpoint(os.path.join(self.saved_models_path,save_best_file_name), monitor='val_acc', verbose=0, save_best_only=True)
        
        print("Start dssm training")
        self.dssm_model.fit_generator(train_generator, 
                                      epochs=epoach, 
                                      steps_per_epoch=train_steps,
                                      shuffle=True,
                                      callbacks = [callback],
                                      verbose=1, 
                                      validation_data=validation_generator,
                                      validation_steps=test_steps)
        
        print("BUILD COLLECTION REPRESENTATION")
        self.build_document_representation()
        print("NORMALZIATION")
        self._normalize_document_collection()

    #create generator from collection data
    class Bag_of_Trigram_Generator(object):
        def __init__(self, dir_name = "bag_of_trigrams"):
            #TODO: Include batch size option            
            path = os.path.join("/backup/pubmed_archive_tokenized",dir_name)
            self.files = map(lambda x:os.path.join(path,x), sorted(os.listdir(path)))
            
        def __iter__(self):
            
            for file in self.files:
                print("Open the file:",file)

                _matrix = sparse.load_npz(file).todense()
                yield _matrix

                del _matrix
                #print("Force garbage collector",gc.collect())

        def __len__(self):
            return len(self.members)
        
        
    def build_document_representation(self):
        
        iter_generator = iter(self.Bag_of_Trigram_Generator())
        
        self.collection_representation = []
        
        for data in iter_generator:
            self.collection_representation.append(self.doc_projection.predict(data, batch_size = 2048, verbose=1))
            del data
            print("Force garbage collector",gc.collect())
            
        self.collection_representation = np.vstack(self.collection_representation)
        """

        gen = self.Bag_of_Trigram_Generator()
        
        def clean_up(batch,logs={}):
            del batch
            print("Force garbage collector",gc.collect())
            
        cleanup_callback = LambdaCallback(on_batch_end=clean_up)
        
        self.collection_representation = self.doc_sub_model.predict_generator(
                                                        iter(gen), 
                                                        steps=len(gen),
                                                        verbose=1,
                                                        callbacks=[cleanup_callback])
        """
        
    class Bag_of_Trigram_Generator_From_Text(object):
        """
        Generator for create bag of trigram text representation
        """
        def __init__(self, data, batch_size = 64,top_k=2500):
            self.data = data
            self.top_k = top_k
            if len(data)<batch_size:
                self.batchs = [0,len(data)]
                self.num_batchs = 1
            else:
                self.batchs = list(range(0,len(data),batch_size))

                self.num_batchs = len(data)//batch_size

                if len(self.batchs) == self.num_batchs:
                    self.batchs.append(len(data))
                else:
                    self.batchs[-1] = len(data)
            
        def __iter__(self):
            
            for i in range(self.num_batchs):
                yield bag_of_trigram(self.data[self.batchs[i]:self.batchs[i+1]])

        def __len__(self):
            return self.num_batchs   
        
    def _predict_process(self, queries):

        queries_representation = self.query_projection.predict(bag_of_trigram(queries), verbose=1)
        
        queries_representation = np.array(queries_representation)
        
        return _post_predict(_sim_cos(self.collection_representation, self.collection_representation_norm, queries_representation), self.top_k)
        
        
        
        
    @staticmethod
    def load(f_name, path = '/backup/saved_models/'):
        file_name = os.path.join(path, f_name)
        
        with open(file_name+"_dssm_metadata.p","rb") as file:
            metadata = pickle.load(file) 
        
        dssm = DSSM(metadata["TRIGRAM_VOC"], build=False)
        dssm.same_q_d_model = metadata["same_q_d_model"]
        dssm.collection_representation = np.load(file_name+"_dssm_doc_emb.npy")
        dssm.trained = metadata["trained"]
        
        if dssm.trained:
            dssm._normalize_document_collection()
           
        dssm.dssm_model = load_model(file_name+"_dssm_model.h5")
        
        siamese_model = dssm.dssm_model.get_layer("siamese_model")

        if dssm.same_q_d_model:
            projection_model = siamese_model.get_layer("projection_model")

            #same model for both
            dssm.query_projection = projection_model
            dssm.doc_projection = projection_model
        else:
            dssm.query_projection = siamese_model.get_layer("query_projection")
            dssm.doc_projection = siamese_model.get_layer("doc_projection")

        return dssm
    
    def save(self, **kwargs):
        
        if "f_name" in kwargs:
            f_name = kwargs.pop("f_name")
        else:
            raise TypeError("f_name must be provided")
        
        if kwargs:
            raise TypeError('Unrecognized keyword arguments: ' + str(kwargs))
            
        file_name = os.path.join(self.saved_models_path, f_name)
        
        #save the document representation
        np.save(file_name+"_dssm_doc_emb",self.collection_representation)
        

        self.dssm_model.save(file_name+"_dssm_model.h5")

        
        #save some metadata
        with open(file_name+"_dssm_metadata.p","wb") as file:
            pickle.dump({"TRIGRAM_VOC":self.TRIGRAM_SIZE,"same_q_d_model":self.same_q_d_model,"trained":self.trained,},file)
        
        
import pickle
import os
import sys
import logging

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import pubmed_data.pubmed_helper as ph

logger = logging.getLogger('mt_bllip')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

fh = logging.FileHandler('mt_bllip.log')
fh.setFormatter(formatter)

logger.addHandler(fh)

logger.info('Test the log file')

map_function = lambda x:x["title"] + " " + x["abstract"] 

pubmed_generator = ph.create_pubmed_collection_generator(map_function)

from keras_new_text import Tokenizer, bllip_stopW_stem_tokenizer

import threading
class TokenizeThread (threading.Thread):
    def __init__(self, threadID, tokenizer, articles):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.thread_progress = 0
        self.tokenizer = tokenizer
        self.articles = articles


    def run(self):
        logger.info("Thread "+str(self.threadID)+" STARTED")
        
        def wrap_tokenizer(text,*kargs):

            if ((self.thread_progress + 1000*self.threadID)%5000)==0:
                logger.info(str(self.threadID)+" :articles tokenized "+str(self.thread_progress))
            self.thread_progress += 1
            return bllip_stopW_stem_tokenizer(text)
        
        self.tokenizer.custom_tokenizer = wrap_tokenizer

        #ALL THREADS RUN THIS
        tokenized = self.tokenizer.texts_to_sequences(self.articles)
        
        file_name = 'bllip_stem_N20_file_' + str(self.threadID)+ '_full_pubmed.p'
        logger.info("save: "+file_name)

        pickle.dump(tokenized,open(os.path.join('/','backup','pubmed_archive_tokenized',file_name),"wb"))
        
        logger.info("Thread "+str(self.threadID)+" ENDED")
        

## Tokanization of the text
threads = []
for i,texts in enumerate(pubmed_generator()):
    tk = ph.load_tokenizer("bllip_stem_small")
    threads.append( TokenizeThread(i,tk,texts))

logger.info("Start for the working threads")
for t in threads:
    t.start()
    
logger.info("Wait for the working threads")
for t in threads:
    t.join()
        

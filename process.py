import numpy as np
from mcerp import PERT  # pip install mcerp  # see https://github.com/tisimst/mcerp/blob/master/mcerp/__init__.py
from laserembeddings import Laser
import math

NUM_TIME_SLOTS = 16
PERT_G = 20

laser = Laser()

# PERT is very slow (50ms per distribution) so we cache a bank of PERT distributions
_num_banks = 100
_xx = np.linspace(start=0, stop=1, num=NUM_TIME_SLOTS)
PERT_BANKS = []
for _pp in np.linspace(0, 1, num=_num_banks):
    if _pp == 0.5:  # some special case that makes g do nothing
        _pp += 0.001
    pert = PERT(low=-0.001, peak=_pp, high=1.001, g=PERT_G, tag=None)
    _yy = pert.rv.pdf(_xx)
    _yy = _yy / sum(_yy)  # hack hack hack
    PERT_BANKS.append(_yy)
    
class Sentence_Configurations():
    def __init__(self):
        self.docs = default_dict(list) #document_id: [sent_vec1,sent_Vec2,sent_Vec3,...]
        self.doc_vecs = default_dict(dict)
        self.tfidf_vec = default_dict(list)
        self.word_freq = default_dict(dict)
        self.doc_ids = []

    def tf(self, word, doc_id, variant):
        sents = self.docs[doc_id]['text']
        word_freq = {}
        word_freq_max = 0
        for s in sents:
            for w in s:
                if w in word_freq:
                    word_freq[w] += 1
                else:
                    word_freq[w] = 1
        
        for w in word_freq.keys():
            if word_freq[w] > word_freq_max:
                word_freq_max = word_freq[w]
        self.word_freq[doc_id] = word_freq
        if variant==2:
            return word_freq[w]
        elif variant==4:
            return 0.4 + 0.6*(word_freq[w]/word_freq_max)
        
    def idf(self, word, variant):
        doc_freq = {}
        for doc_id in self.doc_ids:
            word_freq = self.word_freq[doc_id]
            if word in word_freq.keys():
                doc_freq[word] += 1
            else:
                doc_freq[word] = 0
        
        if variant==4:
            return math.log(1+ len(self.doc_ids)/doc_freq[word])

    def average(self, style="full", document_id):
        #style = [full, top, bottom]
        sent_embs =  self.docs[document_id]
        if style=="full":
            doc_emb = np.average(sent_embs)
            self.doc_vecs[document_id]['full'] = doc_emb
        elif style=="top":
            doc_emb = np.average(sent_embs[:len(sent_embs)//2])
            self.doc_vecs[document_id]['top'] = doc_emb
        
        elif style=="bottom":
            doc_emb = np.average(sent_embs[len(sent_embs)//2:])
            self.doc_vecs[document_id]['bottom'] = doc_emb
        
    def tf_idf_score(self, document_id, variant_tf=4, variant_idf=4):
        for i,sent in enumerate(self.docs[document_id]): #get sentence splitter
            for word in sent: #get word splitter
                score = self.tf(word, variant_tf, document_id)*self.idf(word, variant_idf, document_id)
                self.tfidf_vec[document_id][i] += score
            self.tfidf_vec[document_id][i]/len(sent)
        
            
    def tf_idf_w(self, document_id, variant_tf=4, variant_idf=4):
        self.tf_idf_score(self, document_id, variant_tf, variant_idf)
        sent_embs = self.docs[document_id]
        doc_emb = [self.tfidf_vec[document_id][i]*sent_embs[i] for i in range(len(sent_embs))]
        doc_emb = np.average(doc_emb)
        self.doc_vecs[document_id]['tf_idf_{}_{}'.format(variant_tf, variant_idf)] = doc_emb
        
    
    def pert_doc_vec(self, document_id):
        """
        Standalone example of converting sentence vectors to document vectors
        from https://aclanthology.org/2020.emnlp-main.483.pdf

        Author: Brian Thompson

        """
        sent_vecs = self.docs[document_id]
        sent_counts = len(sent_vecs)
        
        # scale
        sent_weights = 1.0/np.array(sent_counts)

        scaled_sent_vecs = np.multiply(sent_vecs.T, sent_weights).T
        sent_centers = np.linspace(0, 1, len(scaled_sent_vecs))
        sentence_loc_weights = np.zeros((len(sent_centers), NUM_TIME_SLOTS))

        for sent_ii, p in enumerate(sent_centers):
            bank_idx = int(p * (len(PERT_BANKS) - 1))  # find the nearest cached pert distribution
            sentence_loc_weights[sent_ii, :] = PERT_BANKS[bank_idx]
            
        doc_chunk_vec = np.matmul(scaled_sent_vecs.T, sentence_loc_weights).T
        doc_vec = doc_chunk_vec.flatten()
        doc_emb = doc_vec / (np.linalg.norm(doc_vec) + 1e-5)

        self.doc_vecs[document_id]['pert'] = doc_emb
    
    def tf_pert(self, document_id,variant_tf, variant_idf):
        pert_vec = self.doc_vecs[document_id]['pert'] #outputs 1 document vector (pert config) [sent1, ..., sentn]
        self.tf_idf_score(self, document_id, variant_tf, variant_idf) #produced tf_idf vector for doc document_id.
        doc_emb = [self.tfidf_vec[document_id][i]*pert_vec[i] for i in range(len(pert_vec))]
        self.doc_vecs[document_id]['tf_pert'] = doc_emb
        
    def run_all(self, document_id):
        self.average("full",document_id)
        self.average("top",document_id)
        self.average("bottom",document_id)
        self.pert_doc_vec(document_id)
        self.tf_idf_w(document_id,2,4)
        self.tf_idf_w(document_id,4,4)
        self.tf_pert(document_id,2,4)
        self.tf_pert(document_id,4,4)
        

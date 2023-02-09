import numpy as np
from mcerp import PERT  # pip install mcerp  # see https://github.com/tisimst/mcerp/blob/master/mcerp/__init__.py
from laserembeddings import Laser
from sentence_transformers import SentenceTransformer
from collections import defaultdict
import math

NUM_TIME_SLOTS = 16
PERT_G = 20

#PERT is very slow (50ms per distribution) so we cache a bank of PERT distributions
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
#

def meanVector(mat):
    vec = np.array(sum([np.array(mat[i]) for i in range(len(mat))]))
    return vec
    
class Sentence_Configurations:
    def __init__(self):
        self.docs = defaultdict(list) #document_id: [sent_vec1,sent_Vec2,sent_Vec3,...]
        self.doc_vecs = defaultdict(dict)
        self.tfidf_vec = defaultdict(dict)
        self.word_freq = defaultdict(dict)
        self.doc_ids = []
        self.doc_texts = {}

    def tf(self, word, doc_id, variant):
        sents = self.doc_texts[doc_id]
        word_freq = {}
        word_freq_max = 0
        for s in sents:
            for w in s.split(" "):
                if w in word_freq:
                    word_freq[w] += 1
                else:
                    word_freq[w] = 1

        for w in word_freq.keys():
            if word_freq[w] > word_freq_max:
                word_freq_max = word_freq[w]
        self.word_freq[doc_id] = word_freq
        if variant==2:
            if word_freq[w]:
                return word_freq[w]
            else:
                return 1
        elif variant==4:
            if word_freq[w]:
                return 0.4 + 0.6*(word_freq[w]/word_freq_max)
            else:
                return 1
        return 1
    def idf(self, word, variant, doc_id):
        doc_freq = {}
        for doc_id in self.doc_ids:
            word_freq = self.word_freq[doc_id]
            if word in word_freq.keys():
                doc_freq[word] += 1
            else:
                doc_freq[word] = 0

        if variant==4:
            if doc_freq[word]==0:
                return 1
            else:
                return math.log(1+ len(self.doc_ids)/doc_freq[word])
        return 1
    def average(self, style, document_id):
        #style = [full, top, bottom]
        sent_embs =  self.docs[document_id]
        if style=="full":
            doc_emb = meanVector(sent_embs)
            self.doc_vecs[document_id]['full'] = np.array(doc_emb)
        elif style=="top":
            doc_emb = np.pad(meanVector(sent_embs[:len(sent_embs)//2]), (0, (len(sent_embs)//2)), 'constant')
            self.doc_vecs[document_id]['top'] = np.array(doc_emb)

        elif style=="bottom":
            doc_emb = np.pad(meanVector(sent_embs[len(sent_embs)//2:]), ((len(sent_embs)//2), 0), 'constant')
            self.doc_vecs[document_id]['bottom'] = np.array(doc_emb)

    def tf_idf_score(self, document_id, variant_tf, variant_idf):
        for i,sent in enumerate(self.docs[document_id]):
            self.tfidf_vec[document_id][i] = 0
            for word in sent:
                score = self.tf(word,document_id,  variant_tf)*self.idf(word, variant_idf, document_id)
                self.tfidf_vec[document_id][i] += score
            self.tfidf_vec[document_id][i]/len(sent)


    def tf_idf_w(self, document_id, variant_tf, variant_idf):
        self.tf_idf_score(document_id, variant_tf, variant_idf)
        sent_embs = self.docs[document_id]
        doc_emb = [self.tfidf_vec[document_id][j]*sent_embs[i][j] for i in range(len(sent_embs)) for j in range(len(self.tfidf_vec[document_id]))]
        doc_emb = meanVector(doc_emb)
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

        scaled_sent_vecs = np.multiply(np.array(sent_vecs).T, sent_weights).T
        sent_centers = np.linspace(0, 1, len(scaled_sent_vecs))
        sentence_loc_weights = np.zeros((len(sent_centers), NUM_TIME_SLOTS))

        for sent_ii, p in enumerate(sent_centers):
            bank_idx = int(p * (len(PERT_BANKS) - 1))  # find the nearest cached pert distribution
            sentence_loc_weights[sent_ii, :] = PERT_BANKS[bank_idx]

        doc_chunk_vec = np.matmul(scaled_sent_vecs.T, sentence_loc_weights).T
        doc_vec = doc_chunk_vec.flatten()
        doc_emb = doc_vec / (np.linalg.norm(doc_vec) + 1e-5)

        self.doc_vecs[document_id]['pert'] = doc_emb
        self.doc_vecs[document_id]['attn_pert'] = doc_emb

    def tf_pert(self, document_id,variant_tf, variant_idf):
        pert_vec = self.doc_vecs[document_id]['pert'] #outputs 1 document vector (pert config) [sent1, ..., sentn]
        self.tf_idf_score(document_id, variant_tf, variant_idf) #produced tf_idf vector for doc document_id.
        doc_emb = [self.tfidf_vec[document_id][i]*pert_vec[i] for i in range(len(self.tfidf_vec[document_id]))]
        self.doc_vecs[document_id]['tf_pert'] = doc_emb
        self.doc_vecs[document_id]['attn_tf_pert'] = doc_emb

    def run_all(self, document_id):
        self.average("full",document_id)
        self.average("top",document_id)
        self.average("bottom",document_id)
        self.pert_doc_vec(document_id)
#        self.tf_idf_w(document_id,2,4)
#        self.tf_idf_w(document_id,4,4)
        self.tf_pert(document_id,2,4)
        self.tf_pert(document_id,4,4)



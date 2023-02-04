
# Calculate embeddings of MLDoc corpus


import os
import sys
import argparse
from ../process import *


embeddings='./sent_embs/'
doc_embs = './doc_embs/'
datadir='./MLDoc/data/'
doc_ids = []
 i=0
for file in os.listdir(datadir):
    doc_ids.append(i)
    i += 1
    
def generate_doc_vectors(embs='laser'):
    embeddings = embeddings + embs
    i = 0
    for filename in os.listdir(embeddings):
        doc_embeddings = Sentence_Configurations()
        doc_embeddings.__init__()
        doc_embeddings.doc_ids = doc_ids
        doc_embeddings.docs[i] = filename
        doc_embeddings.docs[i]['text'] = filename #maintain same name in MLDoc corpus
        doc_embeddings.run_all(i)
        for emb_type in ['full','top','bottom', 'tf_idf_2_4', 'tf_idf_4_4', 'pert', 'tf_pert', 'attn_pert']:
            doc_dir = doc_embs + emb_type
            f = open("./doc_{}.txt".format(i), w+)
            for j in range(len(doc_embeddings.doc_vecs[i][emb_type])):
                f.write("{}".format(doc_embeddings.doc_vecs[i][emb_type][j]))
                f.write("\n")
            f.close()
        i += 1


# Calculate embeddings of MLDoc corpus


import os
import sys
import argparse
from ../process import *
from laserembeddings import Laser
import math
from sentence_transformers import SentenceTransformer

sbert = SentenceTransformer('all-MiniLM-L6-v2')
labse = SentenceTransformer('sentence-transformers/LaBSE')
laser = Laser()


################################################

datadir = './data/CANTEMIST2020/cantemist/dev-set1/cantemist-coding/txt/'

embeddings='./sent_embs/'
doc_embs = './doc_embs/'
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
        doc_embeddings.docs[i] = [line for line in filename.readlines()]
        if embed_type=="laser":
            doc_embeddings.docs[i] = laser.embed_sentence(doc_embeddings.docs[i], lang=lang_id)
        if embed_type=="sbert";
            doc_embeddings.docs[i] = sbert.encode(doc_embeddings.docs[i])
        if embed_type=="labse":
            doc_embeddings.docs[i] = labse.encode(doc_embeddings.docs[i])
            
        doc_embeddings.docs[i]['text'] = filename #maintain same name in corpus
        doc_embeddings.run_all(i)
        for emb_type in ['full','top','bottom', 'tf_idf_2_4', 'tf_idf_4_4', 'pert', 'tf_pert', 'attn_pert']:
            doc_dir = doc_embs + emb_type
            f = open("{}doc_{}.txt".format(doc_dir,i), w+)
            for j in range(len(doc_embeddings.doc_vecs[i][emb_type])):
                f.write("{}".format(doc_embeddings.doc_vecs[i][emb_type][j]))
                f.write("\n")
            f.close()
        i += 1

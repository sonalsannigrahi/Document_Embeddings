
# Calculate embeddings of MLDoc corpus


import os
import sys
import argparse
from ../process import *


doc_embs = './doc_embs/'
datadir='./MLDoc/data/'

def add_documents():
    i = 0
    for filename in os.listdir(datadir):
        doc_embeddings.doc_ids.append(i)
        doc_embeddings.doc_texts[i] = [line for line in open( datadir + filename).readlines()]
        i += 1
    
def generate_doc_vectors(embs):
    i = 0
    for filename in os.listdir(embeddings):
        doc_embeddings.docs[i] = [line for line in open( datadir + filename).readlines()]
        if embed_type=="laser":
            doc_embeddings.docs[i] = laser.encode(doc_embeddings.docs[i])
        if embed_type=="sbert":
            doc_embeddings.docs[i] = sbert.encode(doc_embeddings.docs[i])
        if embed_type=="labse":
            doc_embeddings.docs[i] = labse.encode(doc_embeddings.docs[i])
        doc_embeddings.run_all(i)
        for emb_type in ['full','top','bottom', 'tf_idf_2_4', 'tf_idf_4_4', 'pert', 'tf_pert', 'attn_pert']:
            doc_dir = doc_embs
            f = open("./doc_{}_{}_{}.txt".format(i, emb_type, embs), w+)
            for j in range(len(doc_embeddings.doc_vecs[i][emb_type])):
                f.write("{}".format(doc_embeddings.doc_vecs[i][emb_type][j]))
                f.write("\n")
            f.close()
        i += 1

if __name__=="__main__":
    doc_embeddings = Sentence_Configurations()
    doc_embeddings.__init__()
    add_documents()
    for emb_type in ['laser', 'sbert', 'labse']:
        generate_doc_vectors(emb_type)

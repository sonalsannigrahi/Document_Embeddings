#
## Calculate embeddings of IDC Code corpus
#
#
import os
import sys
import argparse
from process import *
from laserembeddings import Laser
import math
from sentence_transformers import SentenceTransformer
from collections import defaultdict

laser = Laser()
sbert = SentenceTransformer('all-MiniLM-L6-v2')
labse = SentenceTransformer('sentence-transformers/LaBSE')



###############################################

datadir = './data/CANTEMIST2020/cantemist/dev-set1/cantemist-coding/txt/'
doc_embs = './doc_embs/'
doc_ids = []
#i=0
#for file in os.listdir(datadir):
#    doc_ids.append(i)
#    i += 1

doc_ids = [0]

def add_documents():
    i = 0
    for filename in os.listdir(datadir):
        doc_embeddings.doc_ids.append(i)
        doc_embeddings.doc_texts[i] = [line for line in open( datadir + filename).readlines()]
        i += 1
        
def generate_doc_vectors(embed_type):
    i = 0
    for filename in os.listdir(datadir):
        print(filename)
        doc_embeddings.doc_ids = [0]
        doc_embeddings.docs[i] = [line for line in open( datadir + filename).readlines()]
        if embed_type=="laser":
            doc_embeddings.docs[i] = [[7]*128]*len(doc_embeddings.docs[i])
        if embed_type=="sbert":
            doc_embeddings.docs[i] = sbert.encode(doc_embeddings.docs[i])
        if embed_type=="labse":
            doc_embeddings.docs[i] = labse.encode(doc_embeddings.docs[i])
            
        doc_embeddings.run_all(i)
        for emb_type in ['full','top','bottom', 'pert', 'tf_pert', 'tf_idf_2_4', 'tf_idf_4_4', 'attn_pert', 'attn_tf_pert']:
            doc_dir = doc_embs
            print(doc_embeddings.docs[i])
            print(doc_embeddings.doc_vecs[i])
            with open("{}doc_{}_{}.txt".format(doc_dir,i, emb_type),'w+') as f:
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

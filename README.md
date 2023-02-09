# An Empirical Setup of Multilingual Document Embeddings

In this repository, we provide code for the paper "Are the Best Multilingual Document Embeddings simply Based on Sentence Embeddings?" by Sonal Sannigrahi, Josef van Genabith, Cristina Espana-Bonet in EACL 2023.

## Set Up

The following instructions assume that conda is already installed on your system.

1. Clone this repository, guarding the same directories.
2. Run `conda env create -f environment.yml`
3. Activate the environment `conda activate doc-emb`

## Data Collection 

Due to copyright reasons, we are not able to release datasets. Follow the instructions for the `WMT2016 Bilingual Document Alignment Task`, ` MLDoc (Schwenk and Li, 2018)`, and `ICD Code Classification`

## Run Experiments 

1. Change directories to the task considered, and run `prepare_embs.py` in each directory
2. In case there are library issues, install via pip any missing libraries/modules on your end

### ML-Doc

1. Follow instructions at <href> https://github.com/facebookresearch/MLDoc </href> to process MLDoc Data
2. Once ready, load the LASER Encoders and store data in designated directories 
3. Set directories in the file compute_embs.py
4. Once ready, run sent_classif.py to run the 2 layer MLP classifier. 

### CANTEMIST/ICD Code

1. Run prepare_embs.py to prepare the embeddings 
2. Next, run sentence_classif.py which outputs the accuracy matrix as well (similar to MLDoc)


### WMT2016- Bilingual Doc-Alignment 

1. First, get data from the WMT Task 
2. Run doc_align.py to process the data 
3. Evaluate with WMT-16 code

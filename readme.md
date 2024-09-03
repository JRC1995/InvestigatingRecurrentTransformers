### Credits:
* ```models/modules/xpos_relative_position.py``` is adapted from https://github.com/microsoft/torchscale/tree/main
* ```models/modules/bert_padding.py```  is taken from https://github.com/Dao-AILab/flash-attention
* In ```preprocess/```, ```process_IMBD_lra.py, process_cifar10_lra_sparse,``` and ```process_pathfinder_lra_sparse``` are adapted from https://github.com/google-research/long-range-arena

### Requirements
* pytorch                      1.10.0
* pytorch-lightning            1.9.3
* tqdm                         4.62.3
* tensorflow-datasets          4.5.2
* typing_extensions            4.5.0
* pykeops                      2.1.1
* jsonlines                    2.0.0
* einops                       0.6.0
* torchtext                    0.8.1
* flash-attn                   2.1.1

### Data Setup
* Put the Logical Inference data files (train0,train1,train2,.....test12) (downloaded from https://github.com/yikangshen/Ordered-Memory/tree/master/data/propositionallogic) in data/proplogic/
* Download the ListOps data (along with extrapolation data) from the urls here: https://github.com/facebookresearch/latent-treelstm/blob/master/data/listops/external/urls.txt and put the tsv files in data/listops/
* Run all the make*.py files in data/listops/ to create relevant splits (exact splits used in the paper will be released later) 
* Download LRA (https://github.com/google-research/long-range-arena) dataset
* From LRA dataset put the ListOps basic_test.tsv (LRA test set) in data/listops
* From LRA dataset put the ListOps basic_train.tsv, basic_val.tsv, and basic_test.tsv in data/listops_lra
* From LRA dataset's Retrieval task put new_aan_pairs.train.tsv, new_aan_pairs.eval.tsv, and new_aan_pairs.test.tsv in data/AAN.
* From LRA dataset's Pathfinder task put the folder Pathfinder32 in data/

### Processing Data
* Go to preprocess/ and run each preprocess files to preprocess the corresponding data

### How to train
Train:
```python trian.py --model=[insert model name] -- dataset=[insert dataset name] --times=[insert total runs] --device=[insert device name] --model_type=[classifier/sentence_pair/sentence_pair2]```

* Check argparser.py for exact options. 
* sentence_pair (as model type) is used for sequence matching tasks (logical inference, AAN).
* flipflop (as model type) is used for flipflop language modeling
* classifier for the rest.  
* Generally we use total times as 3.

### Dataset Nomenclature
The nomenclature in the codebase and in the paper are a bit different. We provide a mapping here of the form ([codebase dataset name] == [paper dataset name])

* listopsc2 = ListOps
* proplogic = Logical Inference
* IMDB_lra = Text (LRA)
* AAN_lra = Retrieval (LRA)
* listops_lra = ListOps (LRA)
* cifar10_lra_sparse = Image (LRA)
* pathfinder_lra_sparse = Pathfinder (LRA)

### Model Nomenclature
The nomenclature in the codebase and in the paper are a bit different. We provide a mapping here of the form ([codebase model name] == [paper model name])

* Transformer = Transformer
* UT = UT
* GUT_end = GUT
* GUT_token_end = GUT - Global Halt
* GUT_nogate_end = GUT - Gate
* GUT_notrans_end = GUT - Transition
* TLB = TLB
* GUTLB = GUTLB
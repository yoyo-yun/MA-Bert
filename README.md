## MA-BERT Learning Representation by Incorporating Multi-Attribute Knowledge in Transformers

This repo contains PyTorch deep learning models for personlized classification on three document-level sentiment tasks.

## Usage

- Downloading datasets

  All three datasets, imdb, yelp_2013, and yelp_2014, are followed by Tang. 2015 and available at [here](http://ir.hit.edu.cn/~dytang/paper/acl2015/dataset.7z).

  Unzip for getting datasets listed as the following folders:

  ```
  |-- corpus # 数据集
    |-- imdb
    	|-- ...
    |-- yelp_13
    	|-- ...
    |-- yelp_14
    	|-- ...
  ```

- Running MA-BERT

  ```
  # run: train, val, test
  # dataset: imdb, yelp_13, yelp_14
  # mode: 
  	default: maa (multi-attibute-attention)
  # gpu: 0,1 (a list of gpu ids)
  
  # running code of an instance for IMDB datasets
  python run.py --run train --dataset imdb --mode maa --gpu 0,1 --version fine-tune
  ```

## Noting

All Hyper-Parameters are set in cfgs/config.py.
# The authors' official PyTorch GAT-GAN implementation.

GAT-GAN : A graph-attention-based GAN for time-series generation



## Requirements

To setup the enviroment:

```setup
pip install -r requirements.txt
```

## Datasets

This repository contains implementations of the following real-world datasets.

- Broken rotor data (motor) : https://ieee-dataport.org/keywords/broken-rotor-bar-fault
- Traffic PEMS-SF : https://archive.ics.uci.edu/ml/datasets/PEMS-SF
- MIT-BIH ECH Arrhythmia : https://www.physionet.org/content/mitdb/1.0.0/

## Preprocess

To prerpocess the raw datasets, run the following , the preprocessed data will be stored under the 'preprocessed_data' folder : 

```preprocess
python preprocess --dataset --dir --file_type
```
--dataset : dataset name (ecg/motor/traffic)
--file_type : .csv / .dat
--dir : raw data directory path

## Evaluation (In progress)

To reproduce the numerical results in the paper, run the following to use the models stored in /models depending on sequence length:

Coming soon : Model will be be hosted as a REST API 


## Inference (In progress)

To generate synthetic data , run the following command : 

Coming soon : Model will be be hosted as a REST API 

## Numerical Results

The Frechet transformer distance and predictive scores are saved in the 'output/dataset_name/epoch_no/seq_len/results.txt' file. Additionally, t-SNE, PCA and sample time-series data plots are also saved under the same folder.

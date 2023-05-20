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

```train
python evaluate.py -use_cuda
```
Optionally drop the flag ```-use_cuda``` to run the experiments on CPU.


## Inference (In progress)

To generate synthetic data , run the following command : 

```eval
python inference.py -use_cuda
```
As above, optionally drop the flag ```-use_cuda``` to run the evaluation on CPU.

## Numerical Results

The Frechet transformer distance and predictive scores is saved in the 'output/dataset_name/epoch_no/seq_len/results.txt' file. Additionally, t-SNE, PCA and sample time-series data plots are also saved under the same folder.

Running inference.py will produce the generated data files and their corresponding visualizations in the 'output/evaluations' folder.

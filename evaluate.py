import argparse

import pandas as pd
import torch

def evaluate(args):
    if args.use_cuda:
        device = torch.device('cuda:0')
        print('Using GPU:', device)
    else:
        device = torch.device("cpu")
        print('Using CPU')

    model_path = f'models/'+args.dataset+'/'+str(args.seq_length)+'/'
    data_path = f'preprocessed_data/'

    real_data = pd.read_csv(data_path+args.dataset+'_'+str(args.seq_length)+'.csv').values
    real_data = real_data.reshape(int(real_data.shape[0]/args.seq_length),args.seq_length,real_data.shape[1])
    real_data = torch.tensor(real_data).float().to(device)


    generator_model = torch.load(model_path+'generator.pt')
    print(generator_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="motor")
    parser.add_argument("--seq_length", type=str, default=256)
    parser.add_argument("--use_cuda", type=bool, default= False)
    args = parser.parse_args()
    evaluate(args)
import os
import torch
import argparse
from code.datasets import TrainDataset
from code.datasets import PairDataset
from code.train_model import train_model

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--name')
    p.add_argument('--data_root', default = os.path.join('.', 'data'))
    p.add_argument('--epochs', default=200, type=int)
    p.add_argument('--gf', default=64, type=int)
    p.add_argument('--df', default=64, type=int)
    p.add_argument('--batch_size', default=64, type=int)
    p.add_argument('--noise', default=0.05, type=float)
    p.add_argument('--learn_rate', default=0.0002, type=float)
    p.add_argument('--log_dir', default='./log')
    p.add_argument('--log_rate', default=1, type=int)
    p.add_argument('--log_samples', default=5, type=int)
    p.add_argument('--dump_rate', default=5, type=int)
    p.add_argument('--model', default='generator.pth')
    p.add_argument('--gpu', default=0, type=int)
    args = p.parse_args()

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    train_data = TrainDataset(os.path.join(args.data_root,args.name,'train'))
    valid_data = PairDataset(os.path.join(args.data_root,args.name,'val'))
    model_state = train_model(train_data, valid_data, args.epochs, args.batch_size, 
                args.log_dir, args.log_rate, args.log_samples, args.dump_rate,
                args.gf, args.df, args.noise, args.learn_rate)
    torch.save(model_state,args.model)
    print('Model saved to: ',args.model)
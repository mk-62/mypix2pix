import os
import torch
import argparse
from code.datasets import PairDataset
from code.test_model import test_model

if __name__ == '__main__':   
    p = argparse.ArgumentParser()
    p.add_argument('--name')
    p.add_argument('--data_root', default = os.path.join('.', 'data'))
    p.add_argument('--batch_size', default=64, type=int)
    p.add_argument('--test_dir', default=os.path.join('.', 'test'))
    p.add_argument('--test_count', default=0, type=int)
    p.add_argument('--shuffle', default=1, type=int)
    p.add_argument('--model', default='generator.pth')
    p.add_argument('--gpu', default=0, type=int)
    args = p.parse_args()

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    valid_data = PairDataset(os.path.join(args.data_root, args.name, 'val'))

    model_state = torch.load(args.model)
    test_model(model_state, valid_data, args.batch_size, args.shuffle!=0, args.test_dir, args.test_count)
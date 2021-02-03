import argparse
import matplotlib.pyplot as plt

import torch
import torch.multiprocessing as mp
import torchvision
from torchvision import transforms
import torch.nn as nn
import numpy as np

import NNmodels as nm
import client as clt
import dataLoader as dl
import test as test
import communication_assistant as ca

parser = argparse.ArgumentParser()
parser.add_argument('--n_clients', type=int, default=4, help='')
parser.add_argument('--n_chunks', type=int, default=10, help='')
parser.add_argument('--p_level', type=int, default=10, help='')
parser.add_argument('--batch_size', type=int, default=1, help='')
parser.add_argument('--local_data_ratio', type=float, default=0.01, help='data size ratio each participant has')
parser.add_argument('--n_clients_single_round', type=int, default=5, help='')
parser.add_argument('--n_rounds', type=int, default=10, help='')
parser.add_argument('--model_type', type=nn.Module, default=nm.SimpleDNN)
parser.add_argument('--n_local_epochs', type=int, default=1)
parser.add_argument('--learning_rate', type=float, default=0.01)
'''
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    useGPU = True
    print('학습을 진행하는 기기:', device)
    print('cuda index:', torch.cuda.current_device())
    print('gpu 개수:', torch.cuda.device_count())
    print('graphic name:', torch.cuda.get_device_name())
else:
    device = None
    useGPU = False
    print('학습을 진행하는 기기: CPU')
    '''

      
if __name__ == '__main__':
  opt = parser.parse_args()
  # main_process_model_storage = [None for _ in range(opt.n_clients)]

  train_loader = dl.divideData2Clients(opt.local_data_ratio, opt.batch_size, opt.n_clients, eq_IID=True)

  initialmodel = opt.model_type()


  test_acc_log = [0 for _ in range(opt.n_rounds)]

  clientQs = {i: mp.Queue() for i in range(opt.n_clients)}
  assistantQs = {i: mp.Queue() for i in range(opt.n_clients)}

  # main worker process
  processes = []
  # distillates knowledge from teacher models
  for i in range(opt.n_clients):
    p1 = mp.Process(target=ca.communication_assistant, args=(clientQs[i], assistantQs, i, opt.n_clients))
    p2 = mp.Process(target=clt.make_client, args=(i, train_loader[i], opt.model_type, opt.batch_size, opt.n_clients, clientQs[i], assistantQs[i]))
    p1.start()
    p2.start()
    processes.append(p1)
    processes.append(p2)
  
  for p in processes: p.join()
    
    #test all clients with test dataset
    # for client in clients:
    #   test_acc = test.test(client.model_type, opt.batch_size, client.params)
    #   test_acc_log[i] += test_acc

    # test_acc_log[i] /= opt.n_clients

  # plt.plot(test_acc_log)
  # plt.show()


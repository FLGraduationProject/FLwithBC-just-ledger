import argparse

import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import numpy as np

import NNmodels as nm
import client as clt
import dataLoader as dl
import test as test



import torch.multiprocessing as mp 


parser = argparse.ArgumentParser()
parser.add_argument('--n_clients', type=int, default=5, help='')
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


def KD_trainNtest(client):
  client.KD_train()
  test_acc = test.test(client)

if __name__ == '__main__':
  opt = parser.parse_args()

  train_loader = dl.divideData2Clients(opt.local_data_ratio, opt.batch_size, opt.n_clients, eq_IID=True)

  initialmodel = opt.model_type()

  clients = []
  test_acc_log = [0 for _ in range(opt.n_rounds)]

  # make client
  for i in range(opt.n_clients):
    if i==0:
      clients.append(clt.Client('device-' + str(i), train_loader[i], nm.ComplexDNN, opt.batch_size))

    else:
      clients.append(clt.Client('device-' + str(i), train_loader[i], opt.model_type, opt.batch_size))

  # local train and make local model one time
  for client in clients:
    client.local_train()

  # rounds
  for i in range(opt.n_rounds):
    print(str(i) + " round start")

    # get teacher models from adjacent teacher clients
    for idx, client in enumerate(clients):
      n_teachers = np.random.randint(1, opt.n_clients)
      idx_teachers = np.random.permutation(np.delete(np.arange(opt.n_clients), [idx]))[:n_teachers]
      print(str(idx) + " client teachers are ", idx_teachers)
      client.teachers = [clients[idx_teacher] for idx_teacher in idx_teachers]
      client.get_teacher_models()
    
    # distillates knowledge from teacher models
      # pool = mp.Pool(processes=opt.n_clients)
      # pool.map(KD_trainNtest, clients)
      # pool.close()
      # pool.join()
    processes = []
    for client in clients:    
      client.model.share_memory()
      p = mp.Process(target=KD_trainNtest, args=(client,))
      p.start()
      processes.append(p)
    
    for p in processes: p.join()
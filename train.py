from __future__ import print_function
import os
import sys
directory = os.path.abspath('/scratch/hajorlou/D-VAE/dvae')
directory1 = os.path.abspath('/scratch/hajorlou/D-VAE/src2')

if directory not in sys.path:
    sys.path.append(directory)
if directory1 not in sys.path:
    sys.path.append(directory1)
import math
import pickle
import pdb
import argparse
import random
from tqdm import tqdm
# from shutil import copy
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import scipy.io
from scipy.linalg import qr 
import igraph
from random import shuffle
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from util import *
from models import *
from bayesian_optimization.evaluate_BN import Eval_BN
from src_DAGNN.constants import *
from numpy import linalg as la
from dvae.dagnn import *
import copy
import scipy.io as sio


parser = argparse.ArgumentParser(description='Train Variational Autoencoders for DAGs')
# general settings
parser.add_argument('--data-type', default='ENAS',
                    help='ENAS: ENAS-format CNN structures; BN: Bayesian networks')
parser.add_argument('--data-name', default='final_structures6', help='graph dataset name')
parser.add_argument('--nvt', type=int, default=6, help='number of different node types, \
                    6 for final_structures6, 8 for asia_200k')
parser.add_argument('--save-appendix', default='', 
                    help='what to append to data-name as save-name for results')
parser.add_argument('--save-interval', type=int, default=100, metavar='N',
                    help='how many epochs to wait each time to save model states')
parser.add_argument('--sample-number', type=int, default=20, metavar='N',
                    help='how many samples to generate each time')
parser.add_argument('--no-test', action='store_true', default=False,
                    help='if True, merge test with train, i.e., no held-out set')
parser.add_argument('--reprocess', action='store_true', default=False,
                    help='if True, reprocess data instead of using prestored .pkl data')
parser.add_argument('--keep-old', action='store_true', default=False,
                    help='if True, do not remove any old data in the result folder')
parser.add_argument('--only-test', action='store_true', default=False,
                    help='if True, perform some experiments without training the model')
parser.add_argument('--small-train', action='store_true', default=False,
                    help='if True, use a smaller version of train set')
# model settings
parser.add_argument('--model', default='DCN', help='model to use: DVAE, SVAE, \
                    DVAE_fast, DVAE_BN, SVAE_oneshot, DVAE_GCN,DAGNN,DAGNN_BN,DCN')
parser.add_argument('--load-latest-model', action='store_true', default=False,
                    help='whether to load latest_model.pth')
parser.add_argument('--continue-from', type=int, default=None, 
                    help="from which epoch's checkpoint to continue training")
parser.add_argument('--hs', type=int, default=501, metavar='N',
                    help='hidden size of GRUs')
parser.add_argument('--nz', type=int, default=56, metavar='N',
                    help='number of dimensions of latent vectors z')
parser.add_argument('--bidirectional', action='store_true', default=False,
                    help='whether to use bidirectional encoding')
# optimization settings
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                    help='learning rate (default: 1e-4)')
parser.add_argument('--epochs', type=int, default=100000, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='batch size during training')
parser.add_argument('--infer-batch-size', type=int, default=128, metavar='N',
                    help='batch size during inference')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--all-gpus', action='store_true', default=False,
                    help='use all available GPUs')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

### DAGNN specific arguments
parser.add_argument('--dagnn_layers', type=int, default=2)
parser.add_argument('--dagnn_agg', type=str, default=NA_ATTN_H)
parser.add_argument('--dagnn_out_wx', type=int, default=0, choices=[0, 1])
parser.add_argument('--dagnn_out_pool_all', type=int, default=0, choices=[0, 1])
parser.add_argument('--dagnn_out_pool', type=str, default=P_MAX, choices=[P_ATTN, P_MAX, P_MEAN, P_ADD])
parser.add_argument('--dagnn_dropout', type=float, default=0.0)

parser.add_argument('--clip', default=0, type=float,
                    help='...')
parser.add_argument('--device', type=int, default=0,
                    help='')
parser.add_argument('--res_dir', type=str, default="",
                    help='')
parser.add_argument('--readout', type=str, default='linear', choices=['sum', 'mean', 'max', 'attention', 'set2set','linear'], help='Readout function to apply')

parser.add_argument('--in_dim',type=int ,default= 8)
parser.add_argument('--hid_dim',type=int,default= 501)
# parser.add_argument('--out_dim',type=int,default=70)
parser.add_argument('--n_layers',type=int,default=2)

print(torch.cuda.is_available())

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
np.random.seed(args.seed)
random.seed(args.seed)
print(args)
print(device)

'''Prepare data'''
args.file_dir = os.path.dirname(os.path.realpath('__file__'))
args.res_dir = os.path.join(args.file_dir, 'results1/{}{}'.format(args.data_name, 
                                                                 args.save_appendix))
if not os.path.exists(args.res_dir):
    os.makedirs(args.res_dir) 

pkl_name = os.path.join(args.res_dir, args.data_name + '.pkl')

# check whether to load pre-stored pickle data
# check whether to load pre-stored pickle data
if os.path.isfile(pkl_name) and not args.reprocess:
    with open(pkl_name, 'rb') as f:
        train_data, test_data, graph_args = pickle.load(f)
# otherwise process the raw data and save to .pkl
else:
    # determine data formats according to models, DVAE: igraph, SVAE: string (as tensors)
    # DAGNN
    if "PYG" in args.model or "DAGNN" in args.model:
        input_fmt = 'pyg'
    elif args.model.startswith('DVAE') or args.model.startswith('DCN'):
        input_fmt = 'igraph'
    elif args.model.startswith('SVAE'):
        input_fmt = 'string'
    if args.data_type == 'ENAS':
        train_data, test_data, graph_args = load_ENAS_graphs(args.data_name, n_types=args.nvt,
                                                             fmt=input_fmt)
    elif args.data_type == 'BN':
        train_data, test_data, graph_args = load_BN_graphs(args.data_name, n_types=args.nvt,
                                                           fmt=input_fmt)
    with open(pkl_name, 'wb') as f:
        pickle.dump((train_data, test_data, graph_args), f)

# delete old files in the result directory
remove_list = [f for f in os.listdir(args.res_dir) if not f.endswith(".pkl") and 
        not f.startswith('train_graph') and not f.startswith('test_graph') and
        not f.endswith('.pth')]
for f in remove_list:
    tmp = os.path.join(args.res_dir, f)
    if not os.path.isdir(tmp) and not args.keep_old:
        os.remove(tmp)

# if not args.keep_old:
#     # backup current .py files
#     copy('train.py', args.res_dir)
#     copy('models.py', args.res_dir)
#     copy('util.py', args.res_dir)

# save command line input
cmd_input = 'python ' + ' '.join(sys.argv) + '\n'
with open(os.path.join(args.res_dir, 'cmd_input.txt'), 'a') as f:
    f.write(cmd_input)
print('Command line input: ' + cmd_input + ' is saved.')

# construct train data
if args.no_test:
    train_data = train_data + test_data

if args.small_train:
    train_data = train_data[:100]



if args.model.startswith('DCN'):
    x_train , adj = process_graphs(train_data)

    # print(x_train[0].shape)
    # print(x_train[0])
    myGSOs = GSO_maker(adj)    
    x_train = x_train.to(torch.float32)
    y_train = extract_labels(train_data)

    # x_train = x_train.to('cuda')
    # adj = adj.to('cuda')
    # GSOs = myGSOs.to('cuda') 

    x_test , adj_test = process_graphs(test_data)
    myGSOs_test = GSO_maker(adj_test)    
    x_test = x_test.to(torch.float32)
    y_test = extract_labels(test_data)

    # x_test = x_test.to('cuda')
    # adj_test = adj_test.to('cuda')
    # GSOs_test = myGSOs_test.to('cuda') 


# # ###########
# #     x_test , adj_test = process_graphs(test_data)
# #     test_nx = []
# #     adjacency_test = np.array(adj_test)
# #     adjacency_test_T = np.zeros(adj_test.shape)
# #     tensor_GSOs_test = torch.zeros((len(test_data),8,8,8))
# #     for i in range(len(test_data)):
# #         dag_test = nx.from_numpy_array(np.array(adjacency_test[i]), create_using=nx.DiGraph())
# #         test_nx.append(dag_test)
# #         W_test = la.inv(np.eye(8) - np.array(adjacency_test[i].T))
# #         W_test_inf = la.inv(W_test)
# #         GSOs_test = np.array([(W_test * dagu.compute_Dq(dag_test, i, 8)) @ W_test_inf for i in range(8)])
# #         tensor_GSOs_test[i] = torch.tensor(GSOs_test)

# #     x_test = x_test.to(torch.float32)
# #     x_test = x_test.to('cuda')
# #     GSOs1 = tensor_GSOs_test.to('cuda') 
# #     y_test = extract_labels(test_data)

# #     # x_train, Adj_matrix = process_graphs(train_data)
# #     # labels_tensor = extract_labels(train_data)
# #     # y_train = labels_tensor
# #     # GSOs = GSO_maker(Adj_matrix)
# #     # adj_matrix_T = Adj_matrix.transpose(1, 2)
# #     # GSOs_T = GSO_maker(adj_matrix_T)
# #     # GSOs = GSOs.to(torch.float32)
# #     # GSOs_T = GSOs_T.to(torch.float32)
# #     # y_train = labels_tensor.to(torch.float32)
# #     # x_train = x_train.to(torch.float32)
# #     # x_train = x_train.to('cuda')
# #     # # y_train = y_train.to('cuda')
# #     # # GSOs = GSOs.to('cuda') 
# #     # GSOs_T = GSOs_T.to('cuda')

# #     # x_test, Adj_matrix_test = process_graphs(test_data)
# #     # labels_tensor = extract_labels(test_data)
# #     # y_test = labels_tensor
# #     # GSOs_test = GSO_maker(Adj_matrix_test)
# #     # adj_matrix_test_T = Adj_matrix_test.transpose(1, 2)
# #     # GSOs_test_T = GSO_maker(adj_matrix_test_T)

# #     # GSOs_test = GSOs_test.to(torch.float32) 
# #     # GSOs_test_T = GSOs_test_T.to(torch.float32) 

# #     # y_test = labels_tensor.to(torch.float32)
# #     # x_test = x_test.to(torch.float32)
# #     # x_test = x_test.to('cuda')
# #     # # y_test = y_test.to('cuda')
# #     # # GSOs_test = GSOs_test.to('cuda') 
# #     # GSOs_test_T = GSOs_test_T.to('cuda')
    y_train_np = np.array(y_train)
    y_test_np = np.array(y_test)


# #     # print(x_train.shape)
# #     # print(y_train_np.shape)
# #     # print(GSOs.shape)
# #     # print(len(train_data))

# #     # print(tensor_GSOs.shape)
# #     # print([i for i in range(x_train.shape[0])])

    tuples = [(x_train[i], y_train_np[i], adj[i], train_data[i][0]) for i in range(x_train.shape[0])]
    tuples_test = [(x_test[i], y_test_np[i], adj_test[i], test_data[i][0]) for i in range(x_test.shape[0])]


# #     # tuples = [(x_train[i], y_train[i], GSOs_T[i], train_data[i][0]) for i in range(x_train.shape[0])]
# #     # tuples_test = [(x_test[i], y_test_np[i], GSOs_test_T[i], test_data[i][0]) for i in range(x_test.shape[0])]



# # readout_layer = nn.Linear(501,56)
# # AA = torch.randn((501,100))
# # print(readout_layer(AA))




'''Prepare the model'''
if args.model.startswith("DCN"):
    model = eval('DCN')(graph_args.max_n,
                        graph_args.num_vertex_type,
                        graph_args.START_TYPE,
                        graph_args.END_TYPE,
                        args.in_dim,
                        args.hid_dim,
                        nz = args.nz,
                        K = graph_args.max_n,
                        n_layers= args.n_layers,
                        bidirectional=args.bidirectional)

elif args.model.startswith("DAGNN"):
    model = eval(args.model)(args.nvt + 2, args.hs, args.hs,
                  graph_args.max_n,
                  graph_args.num_vertex_type,
                  graph_args.START_TYPE,
                  graph_args.END_TYPE,
                  hs=args.hs,
                  nz=args.nz,
                  num_nodes=args.nvt+2,
                  agg=args.dagnn_agg,
                  num_layers=args.dagnn_layers,
                  bidirectional=args.bidirectional,
                  out_wx=args.dagnn_out_wx > 0, 
                  out_pool_all=args.dagnn_out_pool_all, 
                  out_pool=args.dagnn_out_pool,
                  dropout=args.dagnn_dropout)
else:
    model = eval(args.model)(
            graph_args.max_n,
            graph_args.num_vertex_type,
            graph_args.START_TYPE,
            graph_args.END_TYPE,
            hs=args.hs,
            nz=args.nz,
            bidirectional=args.bidirectional
            )
    model.mseloss = nn.MSELoss(reduction='sum')


# # optimizer and scheduler
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True)
model.to(device)

if args.all_gpus:
    net = custom_DataParallel(model, device_ids=range(torch.cuda.device_count()))

if args.load_latest_model:
    load_module_state(model, os.path.join(args.res_dir, 'latest_model.pth'))
else:
    if args.continue_from is not None:
        epoch = args.continue_from
        load_module_state(model, os.path.join(args.res_dir, 
                                              'model_checkpoint_{}_{}.pth'.format(args.nz,epoch)))
        load_module_state(optimizer, os.path.join(args.res_dir, 
                                                  'optimizer_checkpoint_{}_{}.pth'.format(args.nz,epoch)))
        load_module_state(scheduler, os.path.join(args.res_dir, 
                                                  'scheduler_checkpoint_{}_{}.pth'.format(args.nz,epoch)))



'''Define some train/test functions'''
def train(epoch):
    model.train()
    train_loss = 0
    recon_loss = 0
    kld_loss = 0
    pred_loss = 0
    shuffle(train_data)
    pbar = tqdm(train_data)
    g_batch = []
    y_batch = []
    for i, (g, y) in enumerate(pbar):
        if args.model.startswith('SVAE'):  # for SVAE, g is tensor
            g = g.to(device)
        g_batch.append(g)
        y_batch.append(y)
        if len(g_batch) == args.batch_size or i == len(train_data) - 1:
            optimizer.zero_grad()
            g_batch = model._collate_fn(g_batch)
            if args.all_gpus: 
                loss = net(g_batch).sum()
                pbar.set_description('Epoch: %d, loss: %0.4f' % (epoch, loss.item()/len(g_batch)))
                recon, kld = 0, 0
            else:
                mu, logvar = model.encode(g_batch)
                loss, recon, kld = model.loss(mu, logvar, g_batch)
                pbar.set_description('Epoch: %d, loss: %0.4f, recon: %0.4f, kld: %0.4f' % (
                                     epoch, loss.item()/len(g_batch), recon.item()/len(g_batch), 
                                     kld.item()/len(g_batch)))
            loss.backward()
            if args.clip > 0:
                torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
            
            train_loss += float(loss)
            recon_loss += float(recon)
            kld_loss += float(kld)
            optimizer.step()
            g_batch = []
            y_batch = []

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_data)))

    return train_loss, recon_loss, kld_loss

def train_DCN(epoch):
    model.train()
    train_loss = 0
    recon_loss = 0
    kld_loss = 0
    pred_loss = 0
    shuffle(tuples)
    pbar = tqdm(tuples)
    g_batch = []
    y_batch = []
    x_batch = []
    adj_mat = []
    for i, (x, y, adj, g) in enumerate(pbar):
        x_batch.append(x)
        y_batch.append(y)
        adj_mat.append(adj)
        g_batch.append(g)
        if len(g_batch) == args.batch_size or i == len(train_data) - 1:
            optimizer.zero_grad()
            xTensor = torch.stack(x_batch, dim=0)
            adj_tens = torch.stack(adj_mat, dim=0)
            mu, logvar = model.encode(xTensor, adj_tens)
            loss, recon, kld = model.loss(mu, logvar, g_batch)
            pbar.set_description('Epoch: %d, loss: %0.4f, recon: %0.4f, kld: %0.4f' % (
                                epoch, loss.item()/len(g_batch), recon.item()/len(g_batch), 
                                kld.item()/len(g_batch)))
            loss.backward()
            train_loss += float(loss)
            recon_loss += float(recon)
            kld_loss += float(kld)
            optimizer.step()
            g_batch = []
            y_batch = []
            x_batch = []
            adj_mat = []

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(tuples)))

    return train_loss, recon_loss, kld_loss


def extract_latent(data):
    model.eval()
    Z = []
    Y = []
    g_batch = []
    for i, (g, y) in enumerate(tqdm(data)):
        if args.model.startswith('SVAE'):
            g_ = g.to(device)
        elif args.model.startswith('DVAE') or args.model.startswith('DAGNN'):
            g_ = copy.deepcopy(g)  
        g_batch.append(g_)
        if len(g_batch) == args.infer_batch_size or i == len(data) - 1:
            g_batch = model._collate_fn(g_batch)
            mu, _ = model.encode(g_batch)
            mu = mu.cpu().detach().numpy()
            Z.append(mu)
            g_batch = []
        Y.append(y)
    return np.concatenate(Z, 0), np.array(Y)


def extract_latent_DCN(data):
    model.eval()
    Z = []
    Y = []
    y_batch = []
    x_batch = []
    shiftop = []
    g_batch = []
    for i, (x, y, GSOs_i, g) in enumerate(tqdm(data)):
        g_ = g.copy()  
        shiftop.append(GSOs_i)
        x_batch.append(x)
        if len(g_batch) == args.infer_batch_size or i == len(data) - 1:
            xTensor = torch.stack(x_batch, dim=0)
            shiftTensor = torch.stack(shiftop, dim=0)
            mu, _ = model.encode(xTensor, shiftTensor)
            mu = mu.cpu().detach().numpy()
            Z.append(mu)
            g_batch = []
            shiftop = []
        Y.append(y)

    return np.concatenate(Z, 0), np.array(Y)


'''Extract latent representations Z'''
def save_latent_representations(epoch):
    if args.model.startswith('DCN'):
        Z_train, Y_train = extract_latent_DCN(tuples)
        Z_test, Y_test = extract_latent_DCN(tuples_test)

        latent_pkl_name = os.path.join(args.res_dir, args.data_name +
                                   '_latent_epoch_{}_{}.pkl'.format(args.nz,epoch))
        latent_mat_name = os.path.join(args.res_dir, args.data_name + 
                                   '_latent_epoch_{}_{}.mat'.format(args.nz,epoch))
        with open(latent_pkl_name, 'wb') as f:
            pickle.dump((Z_train, Y_train, Z_test, Y_test), f)
        print('Saved latent representations to ' + latent_pkl_name)
        
        scipy.io.savemat(latent_mat_name, 
                        mdict={
                            'Z_train': Z_train, 
                            'Z_test': Z_test, 
                            'Y_train': Y_train, 
                            'Y_test': Y_test
                            }
                        )
    else:
        Z_train, Y_train = extract_latent(train_data)
        Z_test, Y_test = extract_latent(test_data)
        latent_pkl_name = os.path.join(args.res_dir, args.data_name +
                                    '_latent_epoch{}.pkl'.format(epoch))
        latent_mat_name = os.path.join(args.res_dir, args.data_name + 
                                    '_latent_epoch{}.mat'.format(epoch))
        with open(latent_pkl_name, 'wb') as f:
            pickle.dump((Z_train, Y_train, Z_test, Y_test), f)
        print('Saved latent representations to ' + latent_pkl_name)
        scipy.io.savemat(latent_mat_name, 
                        mdict={
                            'Z_train': Z_train, 
                            'Z_test': Z_test, 
                            'Y_train': Y_train, 
                            'Y_test': Y_test
                            }
                        )



'''Training begins here'''
min_loss = math.inf  # >= python 3.5
min_loss_epoch = None
loss_name = os.path.join(args.res_dir, 'train_loss.txt')
loss_plot_name = os.path.join(args.res_dir, 'train_loss_plot.pdf')
test_results_name = os.path.join(args.res_dir, 'test_results.txt')
if os.path.exists(loss_name) and not args.keep_old:
    os.remove(loss_name)
print("DEVICE:", device)


start_epoch = args.continue_from if args.continue_from is not None else 0
for epoch in range(start_epoch + 1, args.epochs + 1):
    if args.model.startswith('DCN'):
        train_loss, recon_loss, kld_loss = train_DCN(epoch)
    else:
        train_loss, recon_loss, kld_loss = train(epoch)
        
    with open(loss_name, 'a') as loss_file:
        loss_file.write("{:.2f} {:.2f} {:.2f}\n".format(
            train_loss/len(train_data), 
            recon_loss/len(train_data), 
            kld_loss/len(train_data)
            ))
    scheduler.step(train_loss)
    if epoch % args.save_interval == 0 or epoch == start_epoch + 1 :

        # Z_train, Y_train = extract_latent_DCN(tuples)
        # Z_test, Y_test = extract_latent_DCN(tuples_test)


        print("save current model...")
        model_name = os.path.join(args.res_dir, 'model_checkpoint_{}_{}.pth'.format(args.nz,epoch))
        optimizer_name = os.path.join(args.res_dir, 'optimizer_checkpoint_{}_{}.pth'.format(args.nz,epoch))
        scheduler_name = os.path.join(args.res_dir, 'scheduler_checkpoint_{}_{}.pth'.format(args.nz,epoch))
        torch.save(model.state_dict(), model_name)
        torch.save(optimizer.state_dict(), optimizer_name)
        torch.save(scheduler.state_dict(), scheduler_name)
        print("extract latent representations...")
        save_latent_representations(epoch)


pdb.set_trace()

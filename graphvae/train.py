
import argparse
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
from random import shuffle
import torch
import torch.nn as nn
import torch.nn.init as init 
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter  
from pathlib import Path
from datetime import datetime

import sys 
sys.path.append('.')
import data
from model import GraphVAE
from data import GraphAdjSampler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LR_milestones = [500, 1000]

def build_model(args, max_num_nodes):
    out_dim = max_num_nodes * (max_num_nodes + 1) // 2
    if args.feature_type == 'id':
        input_dim = max_num_nodes
    elif args.feature_type == 'deg':
        input_dim = 1
    elif args.feature_type == 'struct':
        input_dim = 2
    model = GraphVAE(input_dim, 64, 256, max_num_nodes)
    return model

def train(args, dataloader, model, epochs=52):
    date_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_name = f'node{args.max_num_nodes}_lr{args.lr}_epoch{epochs}_{args.dataset}_{args.feature_type}_{date_time}'
    log_dir = Path('runs/graphvae_training') / Path(run_name)
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    optimizer = optim.Adam(list(model.parameters()), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=LR_milestones, gamma=args.lr)
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_idx, data in enumerate(dataloader):
            optimizer.zero_grad()
            features = data['features'].float()
            adj_input = data['adj'].float()

            features = torch.tensor(features).to(DEVICE)
            adj_input = torch.tensor(adj_input).to(DEVICE)
            
            # if args.batch_size > 1:
            #     losses = []
            #     for i in range(features.size(0)):  # Iterate over the batch
            #         loss = model(features[i], adj_input[i])
            #         losses.append(loss)
            #     loss = torch.stack(losses).mean()  # Average the losses
            # else:
            #     loss = model(features, adj_input)
            loss = model(features, adj_input)
            
            loss.backward()
            # Debugging print statement
            # print("Grad NaN in conv1.weight:", torch.isnan(model.conv1.weight.grad).any().item())
            # print("Grad stats:", model.conv1.weight.grad.min().item(), model.conv1.weight.grad.max().item())
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        # Log loss to TensorBoard
        epoch_loss = epoch_loss/len(dataloader)
        print('Epoch: ', epoch, ', Loss: ', loss)
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        
        # save model checkpoint
        if epoch % 5 == 1:
            savedir = Path('model_save/') / Path(run_name)
            savedir.mkdir(parents=True, exist_ok=True)
            fname =  savedir / Path('graphvae_{0}.dat'.format(epoch))
            torch.save(model.state_dict(), fname)

    writer.close()

def arg_parse():
    parser = argparse.ArgumentParser(description='GraphVAE arguments.')
    io_parser = parser.add_mutually_exclusive_group(required=False)
    io_parser.add_argument('--dataset', dest='dataset', 
            help='Input dataset.')

    parser.add_argument('--lr', dest='lr', type=float,
            help='Learning rate.')
    parser.add_argument('--batch_size', dest='batch_size', type=int,
            help='Batch size.')
    parser.add_argument('--num_workers', dest='num_workers', type=int,
            help='Number of workers to load data.')
    parser.add_argument('--max_num_nodes', dest='max_num_nodes', type=int,
            help='Predefined maximum number of nodes in train/test graphs. -1 if determined by \
                  training data.')
    parser.add_argument('--feature', dest='feature_type',
            help='Feature used for encoder. Can be: id, deg')
    parser.add_argument('--save', dest='save',
            help='Save model')

    parser.set_defaults(dataset='grid',
                        feature_type='id',
                        lr=0.0001,
                        batch_size=1,
                        num_workers=4,
                        max_num_nodes=-1)
    return parser.parse_args()

def save_edge_list_to_txt(G, filename):
    with open(filename, 'w') as file:
        for edge in G.edges():
            file.write(f"{edge[0]} {edge[1]}\n")

def build_regular_graphs(graph_dir, max_node=10, graph_rep=10, degree_vec=[2,4,6]):
    graphs = []
    for k in degree_vec: # Number of degrees to test
        for id in range(graph_rep): # Number of graphs to generate
            N = max_node
            G = nx.random_regular_graph(k, N)
            for _ in range(1000): # Number of edges to swap
                G = nx.double_edge_swap(G, nswap=1, max_tries=1000)
            graph_path = graph_dir / Path('regular_{0}_{1}_{2}.txt'.format(N, k, id))
            save_edge_list_to_txt(G, graph_path)
            graphs.append(G)
    print(f'graphs created at {graph_dir}')
    return graphs
        
def main():
    prog_args = arg_parse()

    for dir_path in ['graphs', 'model_save', 'runs']:
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

    print('DEVICE:', DEVICE)
    ### running log
    if prog_args.max_num_nodes != -1:
        max_num_nodes = prog_args.max_num_nodes
    else:
        max_num_nodes = -1

    if prog_args.dataset == 'enzymes':
        graphs= data.Graph_load_batch(min_num_nodes=10, name='ENZYMES')
        num_graphs_raw = len(graphs)

    elif prog_args.dataset == 'grid':
        graphs = []
        for i in range(2,3):
            for j in range(2,3):
                graphs.append(nx.grid_2d_graph(i,j))
        num_graphs_raw = len(graphs)

    elif prog_args.dataset == 'mixed':
        graph_dir = Path('graphs/mixed')
        graphs = []
        for file_path in list(graph_dir.glob('*.txt')):
            graph = nx.read_edgelist(file_path)
            graphs.append(graph)
        print(f'graphs loaded from {graph_dir}')
        num_graphs_raw = len(graphs)

    elif prog_args.dataset == 'regular':
        graph_dir = Path('graphs/regular')
        graph_dir.mkdir(parents=True, exist_ok=True)

        if max_num_nodes > -1:
            graph_dir = graph_dir / Path(f'{max_num_nodes}')
            graph_dir.mkdir(parents=True, exist_ok=True)

        if not any(graph_dir.glob('*.txt')): # if folder is empty
            graphs = build_regular_graphs(
                graph_dir, 
                max_node=max_num_nodes, 
                graph_rep=10, 
                degree_vec=[2,4,6]
                )
            
        else:
            graphs = []
            for file_path in list(graph_dir.glob('*.txt')):
                graph = nx.read_edgelist(file_path)
                graphs.append(graph)
            print(f'graphs loaded from {graph_dir}')
        num_graphs_raw = len(graphs)

    if prog_args.max_num_nodes == -1:
        max_num_nodes = max([graphs[i].number_of_nodes() for i in range(len(graphs))])
    else:
        max_num_nodes = prog_args.max_num_nodes
        # remove graphs with number of nodes greater than max_num_nodes
        graphs = [g for g in graphs if g.number_of_nodes() <= max_num_nodes]

    graphs_len = len(graphs)
    print('Number of graphs removed due to upper-limit of number of nodes: ', 
            num_graphs_raw - graphs_len)
    graphs_test = graphs[int(0.8 * graphs_len):]
    #graphs_train = graphs[0:int(0.8*graphs_len)]
    graphs_train = graphs

    print('total graph num: {}, training set: {}'.format(len(graphs),len(graphs_train)))
    print('max number node: {}'.format(max_num_nodes))

    dataset = GraphAdjSampler(graphs_train, max_num_nodes, features=prog_args.feature_type)
    #sample_strategy = torch.utils.data.sampler.WeightedRandomSampler(
    #        [1.0 / len(dataset) for i in range(len(dataset))],
    #        num_samples=prog_args.batch_size, 
    #        replacement=False)
    dataset_loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=prog_args.batch_size, 
            num_workers=prog_args.num_workers,
            pin_memory=True
    )
    model = build_model(prog_args, max_num_nodes).to(DEVICE)
    train(prog_args, dataset_loader, model)


if __name__ == '__main__':
    main()

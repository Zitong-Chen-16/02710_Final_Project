import numpy as np
import scipy.optimize

import torch
import torch.nn as nn
from torch.nn import BCELoss
from torch import optim
import torch.nn.init as init
import networkx as nx

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# GCN basic operation
class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim).to(DEVICE))
        # self.relu = nn.ReLU()
    def forward(self, x, adj):
        y = torch.matmul(adj, x)
        y = torch.matmul(y,self.weight)
        return y
    
# a deterministic linear output (update: add noise)
class MLP_VAE_plain(nn.Module):
    def __init__(self, h_size, embedding_size, y_size):
        super(MLP_VAE_plain, self).__init__()
        self.encode_11 = nn.Linear(h_size, embedding_size) # mu
        self.encode_12 = nn.Linear(h_size, embedding_size) # lsgms

        self.decode_1 = nn.Linear(embedding_size, embedding_size)
        self.decode_2 = nn.Linear(embedding_size, y_size) # make edge prediction (reconstruct)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))

    def forward(self, h):
        # encoder
        z_mu = self.encode_11(h)
        z_lsgms = self.encode_12(h)
        # prevent inf
        z_mu = torch.clamp(z_mu, min=-10.0, max=10.0)
        z_lsgms = torch.clamp(z_lsgms, min=-10.0, max=10.0)
        # reparameterize
        z_sgm = z_lsgms.mul(0.5).exp_()
        eps = torch.tensor(torch.randn(z_sgm.size())).to(DEVICE)
        z = eps*z_sgm + z_mu
        # decoder
        y = self.decode_1(z)
        y = self.relu(y)
        y = self.decode_2(y)
        return y, z_mu, z_lsgms
    
class GraphVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, max_num_nodes, pool='sum'):
        '''
        Args:
            input_dim: input feature dimension for node.
            hidden_dim: hidden dim for 2-layer gcn.
            latent_dim: dimension of the latent representation of graph.
        '''
        super(GraphVAE, self).__init__()
        self.conv1 = GraphConv(input_dim=input_dim, output_dim=hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 =GraphConv(input_dim=hidden_dim, output_dim=hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.act = nn.ReLU()

        output_dim = max_num_nodes * (max_num_nodes + 1) // 2
        self.vae = MLP_VAE_plain(hidden_dim, latent_dim, output_dim)
        # self.vae = MLP_VAE_plain(input_dim * input_dim, latent_dim, output_dim)
        #self.feature_mlp = model.MLP_plain(latent_dim, latent_dim, output_dim)

        self.max_num_nodes = max_num_nodes
        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.pool = pool

    def recover_adj_lower(self, l):
        # NOTE: Assumes 1 per minibatch
        B = l.size(0)
        adj = torch.zeros(B, self.max_num_nodes, self.max_num_nodes).to(DEVICE)
        triu_idx = torch.triu_indices(self.max_num_nodes, self.max_num_nodes)
        adj[:, triu_idx[0], triu_idx[1]] = l
        # adj[:, torch.triu(torch.ones(self.max_num_nodes, self.max_num_nodes)) == 1] = l
        return adj

    def recover_full_adj_from_lower(self, lower):
        # diag = torch.diag(torch.diag(lower, 0)).to(DEVICE)
        diag = torch.diagonal(lower, dim1=1, dim2=2)
        return lower + lower.transpose(1, 2) - torch.diag_embed(diag)

    def edge_similarity_matrix(self, adj, adj_recon, matching_features,
                matching_features_recon, sim_func):
        B, N, _ = adj.shape
        S = torch.zeros(B, N, N, N, N).to(DEVICE)
        for b in range(B):
            for i in range(N):
                for j in range(N):
                    if i == j:
                        for a in range(N):
                            S[b, i, i, a, a] = adj[b, i, i] * adj_recon[b, a, a] * \
                                            sim_func(matching_features[b, i], matching_features_recon[b, a])
                            # with feature not implemented
                            # if input_features is not None:
                    else:
                        for a in range(N):
                            for c in range(N):
                                if c == a:
                                    continue
                                S[b, i, j, a, c] = adj[b, i, j] * adj[b, i, i] * adj[b, j, j] * \
                                                adj_recon[b, a, c] * adj_recon[b, a, a] * adj_recon[b, c, c]
        return S

    def mpm(self, S, max_iters=50):
        B, N, _, _, _ = S.shape
        x = torch.ones(B, N, N).to(DEVICE) / N
        for _ in range(max_iters):
            x_new = torch.zeros_like(x)
            for b in range(B):
                for i in range(N):
                    for a in range(N):
                        x_new[b, i, a] = x[b, i, a] * S[b, i, i, a, a]
                        pooled = [torch.max(x[b, j, :] * S[b, i, j, a, :])
                                for j in range(N) if j != i]
                        neigh_sim = sum(pooled)
                        x_new[b, i, a] += neigh_sim
            norm = torch.norm(x_new, dim=(1,2), keepdim=True)
            x = x_new / (norm + 1e-8)
        return x 

    def deg_feature_similarity(self, f1, f2):
        return 1 / (torch.abs(f1 - f2) + 1)

    # def permute_adj(self, adj, curr_ind, target_ind):
    #     ''' Permute adjacency matrix.
    #       The target_ind (connectivity) should be permuted to the curr_ind position.
    #     '''
    #     # order curr_ind according to target ind
    #     ind = np.zeros(self.max_num_nodes, dtype=int)
    #     ind[target_ind] = curr_ind
    #     adj_permuted = torch.zeros((self.max_num_nodes, self.max_num_nodes))
    #     adj_permuted[:, :] = adj[ind, :]
    #     adj_permuted[:, :] = adj_permuted[:, ind]
    #     return adj_permuted

    def permute_adj(self, adj, matchings):
        B, N, _ = adj.shape
        permuted = torch.zeros_like(adj)
        for b, (row_ind, col_ind) in enumerate(matchings):
            perm = torch.zeros(N, N).to(DEVICE)
            perm[row_ind, col_ind] = 1.0
            permuted[b] = perm @ adj[b] @ perm.T
        return permuted
    
    def pool_graph(self, x):
        if self.pool == 'max':
            return torch.max(x, dim=1).values
        elif self.pool == 'sum':
            return torch.sum(x, dim=1)

    def forward(self, input_features, adj):
        B, N, _ = adj.shape
        x = self.conv1(input_features, adj)
        x = self.bn1(x.transpose(1, 2)).transpose(1, 2)
        x = self.act(x)
        x = self.conv2(x, adj)
        x = self.bn2(x.transpose(1, 2)).transpose(1, 2)

        # pool over all nodes 
        graph_h = self.pool_graph(x)
        # graph_h = input_features.view(-1, self.max_num_nodes * self.max_num_nodes)
        
        # vae
        h_decode, z_mu, z_lsgms = self.vae(graph_h)
        out_tensor = torch.sigmoid(h_decode)
        recon_adj_lower = self.recover_adj_lower(out_tensor)
        recon_adj_tensor = self.recover_full_adj_from_lower(recon_adj_lower)

        # set matching features be degree
        out_features = torch.sum(recon_adj_tensor, 1)

        adj_data = adj.detach().clone()
        adj_features = torch.sum(adj_data, dim=2)

        S = self.edge_similarity_matrix(adj_data, recon_adj_tensor, adj_features, out_features,
                self.deg_feature_similarity)

        assignment = self.mpm(S)
        matching = self.compute_matching(assignment)
        adj_permuted = self.permute_adj(adj_data, matching)

        # Loss
        triu_idx = torch.triu_indices(self.max_num_nodes, self.max_num_nodes)
        adj_target = adj_permuted[:, triu_idx[0], triu_idx[1]]
        adj_target = torch.clamp(adj_target, 0.0, 1.0) # prevent overflow
        adj_pred = out_tensor
        # assert that adj_pred and adj_target are the same size
        assert adj_pred.size() == adj_target.size(), f"adj_pred size: {adj_pred.size()}, adj_target size: {adj_target.size()}"
        adj_recon_loss = self.adj_recon_loss(adj_target, adj_pred)
        print('recon: ', adj_recon_loss)

        loss_kl = -0.5 * torch.sum(1 + z_lsgms - z_mu.pow(2) - z_lsgms.exp(), dim=1).mean()
        loss_kl /= B * N * N # normalize
        print('kl: ', loss_kl)

        loss = adj_recon_loss + loss_kl

        return loss
    
    def compute_matching(self, assignment):
        B, N, _ = assignment.shape
        matched = []
        for b in range(B):
            row_ind, col_ind = scipy.optimize.linear_sum_assignment(-assignment[b].cpu().detach().numpy())
            matched.append((row_ind, col_ind))
        return matched
    

    
    def forward_test(self, input_features, adj):
        self.max_num_nodes = 4
        adj_data = torch.zeros(self.max_num_nodes, self.max_num_nodes)
        adj_data[:4, :4] = torch.FloatTensor([[1,1,0,0], [1,1,1,0], [0,1,1,1], [0,0,1,1]])
        adj_features = torch.Tensor([2,3,3,2])

        adj_data1 = torch.zeros(self.max_num_nodes, self.max_num_nodes)
        adj_data1 = torch.FloatTensor([[1,1,1,0], [1,1,0,1], [1,0,1,0], [0,1,0,1]])
        adj_features1 = torch.Tensor([3,3,2,2])
        S = self.edge_similarity_matrix(adj_data, adj_data1, adj_features, adj_features1,
                self.deg_feature_similarity)

        # initialization strategies
        init_corr = 1 / self.max_num_nodes
        init_assignment = torch.ones(self.max_num_nodes, self.max_num_nodes) * init_corr
        #init_assignment = torch.FloatTensor(4, 4)
        #init.uniform(init_assignment)
        assignment = self.mpm(init_assignment, S)
        #print('Assignment: ', assignment)

        # matching
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(-assignment.numpy())


        permuted_adj = self.permute_adj(adj_data, row_ind, col_ind)
        print('permuted: ', permuted_adj)

        adj_recon_loss = self.adj_recon_loss(permuted_adj, adj_data1)
        print(adj_data1)
        print('diff: ', adj_recon_loss)

    def adj_recon_loss(self, adj_truth, adj_pred):
        loss = BCELoss()
        return loss(adj_pred,adj_truth)

    # def adj_recon_loss(self, adj_truth, adj_pred):
    #     return F.binary_cross_entropy(adj_truth, adj_pred)

def test_forward_func():
    max_num_nodes = 4 
    G = nx.random_regular_graph(2, max_num_nodes)  

    model = GraphVAE(max_num_nodes, 64, 256, max_num_nodes).to(DEVICE)

    features = torch.tensor(np.identity(max_num_nodes).astype(np.single)).to(DEVICE)
    adj = nx.to_numpy_array(G) + np.identity(G.number_of_nodes())
    adj_padded = np.zeros((max_num_nodes, max_num_nodes))
    adj_padded[:max_num_nodes, :max_num_nodes] = adj
    adj_padded = torch.tensor(adj_padded.astype(np.single)).to(DEVICE)

    loss = model(features, adj_padded)

if __name__ == '__main__':
    test_forward_func()
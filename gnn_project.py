# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 09:59:37 2024

@author: haglers
"""

#
from MAGNA_KGE.MAGNA_KGEncoder import MAGNAKGEncoder
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader

#
from utilities import create_data_graph

def split_interaction_file(data_dir, ratio=1, seed=42):
    M = np.loadtxt(data_dir + "/mat_drug_protein.txt")
    pos_idx = np.array(np.where(M == 1)).T
    neg_idx = np.array(np.where(M == 0)).T
    _rng = np.random.RandomState(seed)
    neg_sampled_idx = _rng.choice(neg_idx.shape[0], int(pos_idx.shape[0] * ratio), replace=False)
    neg_idx = neg_idx[neg_sampled_idx]

    train_pos_idx, test_pos_idx = train_test_split(pos_idx, test_size=0.2, random_state=seed)
    train_neg_idx, test_neg_idx = train_test_split(neg_idx, test_size=0.2, random_state=seed)

    train_idx = np.concatenate((train_pos_idx, train_neg_idx), axis=0)
    train_y = np.concatenate((np.ones(train_pos_idx.shape[0]), np.zeros(train_neg_idx.shape[0])))
    train = []
    train_idx = train_idx.tolist()
    train_y = train_y.tolist()
    for i in range(len(train_idx)):
        x = train_idx[i]
        x.extend([train_y[i]])
        train.append(x)
    
    #train_P = np.zeros(M.shape)
    #train_P[train_idx[:, 0], train_idx[:, 1]] = 1

    test_idx = np.concatenate((test_pos_idx, test_neg_idx))
    test_y = np.concatenate((np.ones(test_pos_idx.shape[0]), np.zeros(test_neg_idx.shape[0])))
      
    test = []
    test_idx = test_idx.tolist()
    test_y = test_y.tolist()
    for i in range(len(test_idx)):
        x = test_idx[i]
        x.extend([test_y[i]])
        test.append(x)

    #return train_P, test_idx, test_y
    return train, test

def test_loop(dataloader, graph, model, loss_fn, current_performance):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X in dataloader:
            pred = torch.round(model(graph, X[:2]))
            test_loss += loss_fn(pred, X[2]).item()
            correct += (pred == X[2]).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"New Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n", flush=True)
    if correct > current_performance[0]:
        torch.save(model.state_dict(), 'model/model.pth')
        current_performance[0] = correct
        current_performance[1] = test_loss
    print(f"Current Test Error: \n Accuracy: {(100*current_performance[0]):>0.1f}%, Avg loss: {current_performance[1]:>8f} \n", flush=True)
    return current_performance

def train_loop(dataloader, graph, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, X in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(graph, X[:2])
        loss = loss_fn(pred, X[2].float())
    
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
        if batch % 10 == 0:
            loss, current = loss.item(), batch * batch_size + len(X[:2])
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]", flush=True)

class Model(nn.Module):
    
    def __init__(self, num_drugs, num_proteins, ntriples, num_nodes, num_edges):
        super().__init__()
        num_heads = 8
        num_layers = 3
        in_ent_dim = 256
        in_rel_dim = 256
        topk = 10
        topk_type = 'local'
        alpha = 0.05
        hidden_dim = 256 #128
        hop_num = 2
        input_drop = 0
        feat_drop = 0
        attn_drop = 0
        edge_drop = 0.1
        negative_slope = 0.2
        self.num_drugs = num_drugs
        self.num_proteins = num_proteins
        self.entity_embedding = \
            nn.Parameter(torch.zeros(num_nodes, in_ent_dim), requires_grad=True)
        self.relation_embedding = \
            nn.Parameter(torch.zeros(num_edges, in_rel_dim), requires_grad=True)
        self.ent_map = nn.Linear(in_ent_dim, in_ent_dim, bias=False)
        self.rel_map = nn.Linear(in_rel_dim, in_rel_dim, bias=False)
        self.dag_entity_encoder = MAGNAKGEncoder(num_layers,
                                                 in_ent_dim,
                                                 in_rel_dim,
                                                 topk,
                                                 num_heads,
                                                 alpha,
                                                 hidden_dim,
                                                 hop_num,
                                                 input_drop,
                                                 feat_drop,
                                                 attn_drop,
                                                 topk_type,
                                                 edge_drop,
                                                 negative_slope,
                                                 ntriples)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
            
    def forward(self, graph, x):
        entity_embedder = self.ent_map(self.entity_embedding)
        relation_embedder = self.rel_map(self.relation_embedding)
        entity_embedder = \
            self.dag_entity_encoder(graph, entity_embedder, relation_embedder)
        H0 = entity_embedder[0]
        H0 = torch.transpose(H0, 0, 1)
        a = torch.zeros(self.num_drugs, len(x[0]))
        b = torch.zeros(self.num_proteins, len(x[0]))
        for i in range(len(x[0])):
            a[x[0][i], i] = 1
            b[x[1][i], i] = 1
        dim = graph.number_of_nodes() - a.size(0) - b.size(0)
        c = torch.zeros(dim, len(x[0]))
        h = torch.cat((a, b, c), axis=0)
        X0 = torch.matmul(H0, h)
        X0 = torch.transpose(X0, 0, 1)
        X = self.classifier(X0)
        X = torch.squeeze(X, 1)
        return X
    
if __name__ == '__main__':
    data_dir = 'pydtinet_data'
    learning_rate = 5e-5
    beta1 = 0.90
    beta2 = 0.99
    weight_decay = 0.01
    batch_size = 128 #64
    epochs = 40
    drug_similarity_threshold = None
    protein_similarity_threshold = None
    restarts = 10
    graph, num_drugs, num_proteins, ntriples = \
        create_data_graph(data_dir,
                          drug_similarity_threshold,
                          protein_similarity_threshold,
                          'mat_drug_drug.txt',
                          'mat_drug_protein.txt',
                          'mat_protein_protein.txt',
                          'mat_drug_disease.txt',
                          'mat_protein_disease.txt',
                          'mat_drug_se.txt',
                          'Similarity_Matrix_Drugs.txt',
                          'Similarity_Matrix_Proteins.txt')
    train, test = split_interaction_file(data_dir)
    train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test, batch_size=batch_size)
    loss_fn = nn.CrossEntropyLoss()
    current_performance = [ 0, 0 ]
    for s in range(restarts):
        model = Model(num_drugs, num_proteins, ntriples, graph.number_of_nodes(),
                      graph.number_of_edges())
        optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)
        for t in range(epochs):
            print(f"Epoch {s+1} / {t+1}\n-------------------------------")
            train_loop(train_dataloader, graph, model, loss_fn, optimizer)
            current_performance = \
                test_loop(test_dataloader, graph, model, loss_fn, current_performance)
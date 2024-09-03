# -*- coding: utf-8 -*-
"""
Created on Fri May 10 12:47:57 2024

@author: haglers
"""

#
import dgl
import numpy as np
import torch

def create_data_graph(data_dir, drug_similarity_threshold,
                      protein_similarity_threshold, drug_drug_file,
                      drug_protein_file, protein_protein_file,
                      drug_disease_file, protein_disease_file, drug_se_file,
                      drug_similarity_file, protein_similarity_file):
    disease_self_label = 0
    drug_disease_label = 1
    drug_drug_label = 2
    drug_protein_label = 3
    drug_se_label = 4
    drug_self_label = 5
    protein_protein_label = 6
    protein_disease_label = 7
    protein_self_label = 8
    se_self_label = 9
    drug_similarity_label = 10
    protein_similarity_label = 11
    drug_drug_mat = \
        np.array(np.loadtxt(data_dir + "/" + drug_drug_file))
    drug_protein_mat = \
        np.array(np.loadtxt(data_dir + "/" + drug_protein_file))
    protein_protein_mat = \
        np.array(np.loadtxt(data_dir + "/" + protein_protein_file))
    drug_disease_mat = \
        np.array(np.loadtxt(data_dir + "/" + drug_disease_file))
    protein_disease_mat = \
        np.array(np.loadtxt(data_dir + "/" + protein_disease_file))
    drug_se_mat = \
        np.array(np.loadtxt(data_dir + "/" + drug_se_file))
    drug_similarity_mat = \
        np.array(np.loadtxt(data_dir + "/" + drug_similarity_file))
    protein_similarity_mat = \
        np.array(np.loadtxt(data_dir + "/" + protein_similarity_file))
    disease_disease_zeros = np.zeros((len(drug_disease_mat[0]),
                                      len(drug_disease_mat[0])))
    disease_se_zeros = np.zeros((len(drug_disease_mat[0]),
                                 len(drug_se_mat[0])))
    drug_disease_zeros = \
        np.zeros((len(drug_disease_mat), len(drug_disease_mat[0])))
    drug_drug_zeros = np.zeros((len(drug_drug_mat), len(drug_drug_mat[0])))
    drug_protein_zeros = \
        np.zeros((len(drug_protein_mat), len(drug_protein_mat[0])))
    drug_se_zeros = np.zeros((len(drug_se_mat), len(drug_se_mat[0])))
    protein_disease_zeros = \
        np.zeros((len(protein_disease_mat), len(protein_disease_mat[0])))
    protein_protein_zeros = \
        np.zeros((len(protein_protein_mat), len(protein_protein_mat[0])))
    protein_se_zeros = \
        np.zeros((len(protein_protein_mat), len(drug_se_mat[0])))
    se_se_zeros = np.zeros((len(drug_se_mat[0]), len(drug_se_mat[0])))
    disease_identity = np.identity(len(drug_disease_mat[0]))
    drug_identity = np.identity(len(drug_drug_mat))
    protein_identity = np.identity(len(protein_protein_mat))
    se_identity = np.identity(len(drug_se_mat[0]))
    g = dgl.DGLGraph()
    if False:
        m0 = np.concatenate((drug_drug_mat, drug_protein_zeros), axis=1)
        m1 = np.concatenate((np.transpose(drug_protein_zeros),
                             protein_protein_zeros), axis=1)
        m = np.concatenate((m0, m1), axis=0)
        src, dst = np.nonzero(m)
        g.add_nodes(len(m))
        g.add_edges(src, dst)
        elabel_0 = drug_drug_label * torch.ones(len(src), 1, dtype=torch.long)
        
        m0 = np.concatenate((drug_drug_zeros, drug_protein_mat), axis=1)
        m1 = np.concatenate((np.transpose(drug_protein_mat),
                             protein_protein_zeros), axis=1)
        m = np.concatenate((m0, m1), axis=0)
        src, dst = np.nonzero(m)
        g.add_edges(src, dst)
        elabel_1 = \
            drug_protein_label * torch.ones(len(src), 1, dtype=torch.long)
        
        m0 = np.concatenate((drug_drug_zeros, drug_protein_zeros), axis=1)
        m1 = np.concatenate((np.transpose(drug_protein_zeros), 
                             protein_protein_mat), axis=1)
        m = np.concatenate((m0, m1), axis=0)
        src, dst = np.nonzero(m)
        g.add_edges(src, dst)
        elabel_2 = \
            protein_protein_label * torch.ones(len(src), 1, dtype=torch.long)
        
        m0 = np.concatenate((drug_identity, drug_protein_zeros), axis=1)
        m1 = np.concatenate((np.transpose(drug_protein_zeros), 
                             protein_protein_zeros), axis=1)
        m = np.concatenate((m0, m1), axis=0)
        src, dst = np.nonzero(m)
        g.add_edges(src, dst)
        elabel_3 = drug_self_label * torch.ones(len(src), 1, dtype=torch.long)
        
        m0 = np.concatenate((drug_drug_zeros, drug_protein_zeros), axis=1)
        m1 = np.concatenate((np.transpose(drug_protein_zeros), 
                             protein_identity), axis=1)
        m = np.concatenate((m0, m1), axis=0)
        src, dst = np.nonzero(m)
        g.add_edges(src, dst)
        elabel_4 = \
            protein_self_label * torch.ones(len(src), 1, dtype=torch.long)
        
        g.edata['e_label'] = torch.cat([elabel_0, elabel_1, elabel_2,
                                        elabel_3, elabel_4], dim=0)
    else:
        
        #
        elabel_list = []
        elabel_eye_list = []
        
        # drug_drug_mat
        m0 = np.concatenate((drug_drug_mat, drug_protein_zeros,
                             drug_disease_zeros, drug_se_zeros), axis=1)
        m1 = np.concatenate((np.transpose(drug_protein_zeros),
                             protein_protein_zeros, protein_disease_zeros,
                             protein_se_zeros), axis=1)
        m2 = np.concatenate((np.transpose(drug_disease_zeros), 
                             np.transpose(protein_disease_zeros),
                             disease_disease_zeros, disease_se_zeros), axis=1)
        m3 = np.concatenate((np.transpose(drug_se_zeros),
                             np.transpose(protein_se_zeros),
                             np.transpose(disease_se_zeros), se_se_zeros),
                            axis=1)
        m = np.concatenate((m0, m1, m2, m3), axis=0)
        g.add_nodes(len(m))
        m = np.triu(m, 1)
        src, dst = np.nonzero(m)
        src_u = src
        dst_u = dst
        src_l = dst
        dst_l = src
        elabel_list.append(drug_drug_label * torch.ones(len(src), 1, dtype=torch.long))
        if drug_similarity_threshold is not None:
            x = drug_similarity_mat - drug_identity
            m0 = np.concatenate((x, drug_protein_zeros,
                                 drug_disease_zeros, drug_se_zeros), axis=1)
            m1 = np.concatenate((np.transpose(drug_protein_zeros),
                                 protein_protein_zeros, protein_disease_zeros,
                                 protein_se_zeros), axis=1)
            m2 = np.concatenate((np.transpose(drug_disease_zeros), 
                                 np.transpose(protein_disease_zeros),
                                 disease_disease_zeros, disease_se_zeros), axis=1)
            m3 = np.concatenate((np.transpose(drug_se_zeros),
                                 np.transpose(protein_se_zeros),
                                 np.transpose(disease_se_zeros), se_se_zeros),
                                axis=1)
            m = np.concatenate((m0, m1, m2, m3), axis=0)
            idxs = np.where(m < drug_similarity_threshold)
            m[idxs[0], idxs[1]] = 0
            idxs = np.where(m != 0)
            m[idxs[0], idxs[1]] = 1
            m = np.triu(m, 1)
            src, dst = np.nonzero(m)
            print(len(src))
            src_u = np.concatenate((src_u, src), axis=0)
            dst_u = np.concatenate((dst_u, dst), axis=0)
            src_l = np.concatenate((src_l, dst), axis=0)
            dst_l = np.concatenate((dst_l, src), axis=0)
            elabel_list.append(drug_similarity_label * torch.ones(len(src), 1, dtype=torch.long))
        
        # drug_protein_mat
        m0 = np.concatenate((drug_drug_zeros, drug_protein_mat,
                             drug_disease_zeros, drug_se_zeros), axis=1)
        m1 = np.concatenate((np.transpose(drug_protein_mat),
                             protein_protein_zeros, protein_disease_zeros,
                             protein_se_zeros), axis=1)
        m2 = np.concatenate((np.transpose(drug_disease_zeros), 
                             np.transpose(protein_disease_zeros),
                             disease_disease_zeros, disease_se_zeros), axis=1)
        m3 = np.concatenate((np.transpose(drug_se_zeros),
                             np.transpose(protein_se_zeros),
                             np.transpose(disease_se_zeros), se_se_zeros),
                            axis=1)
        m = np.concatenate((m0, m1, m2, m3), axis=0)
        m = np.triu(m, 1)
        src, dst = np.nonzero(m)
        src_u = np.concatenate((src_u, src), axis=0)
        dst_u = np.concatenate((dst_u, dst), axis=0)
        src_l = np.concatenate((src_l, dst), axis=0)
        dst_l = np.concatenate((dst_l, src), axis=0)
        elabel_list.append(drug_protein_label * torch.ones(len(src), 1, dtype=torch.long))
            
        # protein_protein_mat
        m0 = np.concatenate((drug_drug_zeros, drug_protein_zeros,
                             drug_disease_zeros, drug_se_zeros), axis=1)
        m1 = np.concatenate((np.transpose(drug_protein_zeros),
                             protein_protein_mat, protein_disease_zeros,
                             protein_se_zeros), axis=1)
        m2 = np.concatenate((np.transpose(drug_disease_zeros), 
                             np.transpose(protein_disease_zeros),
                             disease_disease_zeros, disease_se_zeros), axis=1)
        m3 = np.concatenate((np.transpose(drug_se_zeros),
                             np.transpose(protein_se_zeros),
                             np.transpose(disease_se_zeros), se_se_zeros),
                            axis=1)
        m = np.concatenate((m0, m1, m2, m3), axis=0)
        m = np.triu(m, 1)
        src, dst = np.nonzero(m)
        src_u = np.concatenate((src_u, src), axis=0)
        dst_u = np.concatenate((dst_u, dst), axis=0)
        src_l = np.concatenate((src_l, dst), axis=0)
        dst_l = np.concatenate((dst_l, src), axis=0)
        elabel_list.append(protein_protein_label * torch.ones(len(src), 1, dtype=torch.long))
        if protein_similarity_threshold is not None:
            x = protein_similarity_mat - protein_identity
            m0 = np.concatenate((drug_drug_zeros, drug_protein_zeros,
                                 drug_disease_zeros, drug_se_zeros), axis=1)
            m1 = np.concatenate((np.transpose(drug_protein_zeros),
                                 x, protein_disease_zeros,
                                 protein_se_zeros), axis=1)
            m2 = np.concatenate((np.transpose(drug_disease_zeros), 
                                 np.transpose(protein_disease_zeros),
                                 disease_disease_zeros, disease_se_zeros), axis=1)
            m3 = np.concatenate((np.transpose(drug_se_zeros),
                                 np.transpose(protein_se_zeros),
                                 np.transpose(disease_se_zeros), se_se_zeros),
                                axis=1)
            m = np.concatenate((m0, m1, m2, m3), axis=0)
            idxs = np.where(m < 100 * protein_similarity_threshold)
            m[idxs[0], idxs[1]] = 0
            idxs = np.where(m != 0)
            m[idxs[0], idxs[1]] = 1
            m = np.triu(m, 1)
            src, dst = np.nonzero(m)
            print(len(src))
            src_u = np.concatenate((src_u, src), axis=0)
            dst_u = np.concatenate((dst_u, dst), axis=0)
            src_l = np.concatenate((src_l, dst), axis=0)
            dst_l = np.concatenate((dst_l, src), axis=0)
            elabel_list.append(protein_similarity_label * torch.ones(len(src), 1, dtype=torch.long))
           
        # drug_disease_mat
        m0 = np.concatenate((drug_drug_zeros, drug_protein_zeros,
                             drug_disease_mat, drug_se_zeros), axis=1)
        m1 = np.concatenate((np.transpose(drug_protein_zeros),
                             protein_protein_zeros, protein_disease_zeros,
                             protein_se_zeros), axis=1)
        m2 = np.concatenate((np.transpose(drug_disease_mat), 
                             np.transpose(protein_disease_zeros),
                             disease_disease_zeros, disease_se_zeros), axis=1)
        m3 = np.concatenate((np.transpose(drug_se_zeros),
                             np.transpose(protein_se_zeros),
                             np.transpose(disease_se_zeros), se_se_zeros),
                            axis=1)
        m = np.concatenate((m0, m1, m2, m3), axis=0)
        m = np.triu(m, 1)
        src, dst = np.nonzero(m)
        src_u = np.concatenate((src_u, src), axis=0)
        dst_u = np.concatenate((dst_u, dst), axis=0)
        src_l = np.concatenate((src_l, dst), axis=0)
        dst_l = np.concatenate((dst_l, src), axis=0)
        elabel_list.append(drug_disease_label * torch.ones(len(src), 1, dtype=torch.long))
           
        # drug_se_mat
        m0 = np.concatenate((drug_drug_zeros, drug_protein_zeros,
                             drug_disease_zeros, drug_se_mat), axis=1)
        m1 = np.concatenate((np.transpose(drug_protein_zeros),
                             protein_protein_zeros, protein_disease_zeros,
                             protein_se_zeros), axis=1)
        m2 = np.concatenate((np.transpose(drug_disease_zeros), 
                             np.transpose(protein_disease_zeros),
                             disease_disease_zeros, disease_se_zeros), axis=1)
        m3 = np.concatenate((np.transpose(drug_se_mat),
                             np.transpose(protein_se_zeros),
                             np.transpose(disease_se_zeros), se_se_zeros),
                            axis=1)
        m = np.concatenate((m0, m1, m2, m3), axis=0)
        m = np.triu(m, 1)
        src, dst = np.nonzero(m)
        src_u = np.concatenate((src_u, src), axis=0)
        dst_u = np.concatenate((dst_u, dst), axis=0)
        src_l = np.concatenate((src_l, dst), axis=0)
        dst_l = np.concatenate((dst_l, src), axis=0)
        elabel_list.append(drug_se_label * torch.ones(len(src), 1, dtype=torch.long))
        
        # protein_disease_mat
        m0 = np.concatenate((drug_drug_zeros, drug_protein_zeros,
                             drug_disease_zeros, drug_se_zeros), axis=1)
        m1 = np.concatenate((np.transpose(drug_protein_zeros),
                             protein_protein_zeros, protein_disease_mat,
                             protein_se_zeros), axis=1)
        m2 = np.concatenate((np.transpose(drug_disease_zeros), 
                             np.transpose(protein_disease_mat),
                             disease_disease_zeros, disease_se_zeros), axis=1)
        m3 = np.concatenate((np.transpose(drug_se_zeros),
                             np.transpose(protein_se_zeros),
                             np.transpose(disease_se_zeros), se_se_zeros),
                            axis=1)
        m = np.concatenate((m0, m1, m2, m3), axis=0)
        m = np.triu(m, 1)
        src, dst = np.nonzero(m)
        src_u = np.concatenate((src_u, src), axis=0)
        dst_u = np.concatenate((dst_u, dst), axis=0)
        src_l = np.concatenate((src_l, dst), axis=0)
        dst_l = np.concatenate((dst_l, src), axis=0)
        elabel_list.append(protein_disease_label * torch.ones(len(src), 1, dtype=torch.long))
        
        # drug_identity
        m0 = np.concatenate((drug_identity, drug_protein_zeros,
                             drug_disease_zeros, drug_se_zeros), axis=1)
        m1 = np.concatenate((np.transpose(drug_protein_zeros),
                             protein_protein_zeros, protein_disease_zeros,
                             protein_se_zeros), axis=1)
        m2 = np.concatenate((np.transpose(drug_disease_zeros), 
                             np.transpose(protein_disease_zeros),
                             disease_disease_zeros, disease_se_zeros), axis=1)
        m3 = np.concatenate((np.transpose(drug_se_zeros),
                             np.transpose(protein_se_zeros),
                             np.transpose(disease_se_zeros), se_se_zeros),
                            axis=1)
        m = np.concatenate((m0, m1, m2, m3), axis=0)
        src, dst = np.nonzero(m)
        src_eye = src
        dst_eye = dst
        elabel_eye_list.append(drug_self_label * torch.ones(len(src), 1, dtype=torch.long))
        
        # protein_idetity
        m0 = np.concatenate((drug_drug_zeros, drug_protein_zeros,
                             drug_disease_zeros, drug_se_zeros), axis=1)
        m1 = np.concatenate((np.transpose(drug_protein_zeros),
                             protein_identity, protein_disease_zeros,
                             protein_se_zeros), axis=1)
        m2 = np.concatenate((np.transpose(drug_disease_zeros), 
                             np.transpose(protein_disease_zeros),
                             disease_disease_zeros, disease_se_zeros), axis=1)
        m3 = np.concatenate((np.transpose(drug_se_zeros),
                             np.transpose(protein_se_zeros),
                             np.transpose(disease_se_zeros), se_se_zeros),
                            axis=1)
        m = np.concatenate((m0, m1, m2, m3), axis=0)
        src, dst = np.nonzero(m)
        src_eye = np.concatenate((src_eye, src), axis=0)
        dst_eye = np.concatenate((dst_eye, dst), axis=0)
        elabel_eye_list.append(protein_self_label * torch.ones(len(src), 1, dtype=torch.long))
        
        # disease_identity
        m0 = np.concatenate((drug_drug_zeros, drug_protein_zeros,
                             drug_disease_zeros, drug_se_zeros), axis=1)
        m1 = np.concatenate((np.transpose(drug_protein_zeros),
                             protein_protein_zeros, protein_disease_zeros,
                             protein_se_zeros), axis=1)
        m2 = np.concatenate((np.transpose(drug_disease_zeros), 
                             np.transpose(protein_disease_zeros),
                             disease_identity, disease_se_zeros), axis=1)
        m3 = np.concatenate((np.transpose(drug_se_zeros),
                             np.transpose(protein_se_zeros),
                             np.transpose(disease_se_zeros), se_se_zeros),
                            axis=1)
        m = np.concatenate((m0, m1, m2, m3), axis=0)
        src, dst = np.nonzero(m)
        src_eye = np.concatenate((src_eye, src), axis=0)
        dst_eye = np.concatenate((dst_eye, dst), axis=0)
        elabel_eye_list.append(disease_self_label * torch.ones(len(src), 1, dtype=torch.long))
            
        # se_identity
        m0 = np.concatenate((drug_drug_zeros, drug_protein_zeros,
                             drug_disease_zeros, drug_se_zeros), axis=1)
        m1 = np.concatenate((np.transpose(drug_protein_zeros),
                             protein_protein_zeros, protein_disease_zeros,
                             protein_se_zeros), axis=1)
        m2 = np.concatenate((np.transpose(drug_disease_zeros), 
                             np.transpose(protein_disease_zeros),
                             disease_disease_zeros, disease_se_zeros), axis=1)
        m3 = np.concatenate((np.transpose(drug_se_zeros),
                             np.transpose(protein_se_zeros),
                             np.transpose(disease_se_zeros), se_identity),
                            axis=1)
        m = np.concatenate((m0, m1, m2, m3), axis=0)
        src, dst = np.nonzero(m)
        src_eye = np.concatenate((src_eye, src), axis=0)
        dst_eye = np.concatenate((dst_eye, dst), axis=0)
        elabel_eye_list.append(se_self_label * torch.ones(len(src), 1, dtype=torch.long))
        
        src = src_u
        src = np.concatenate((src, src_l), axis=0)
        src = np.concatenate((src, src_eye), axis=0)
        
        dst = dst_u
        dst = np.concatenate((dst, dst_l), axis=0)
        dst = np.concatenate((dst, dst_eye), axis=0)
        
        elabel = torch.cat(elabel_list, dim=0)
        ntriples = elabel.size(0)
        
        elabel_list.extend(elabel_list)
        elabel_list.extend(elabel_eye_list)
        elabel = torch.cat(elabel_list, dim=0)
        
        g.add_edges(src, dst)
        g.edata['e_label'] = elabel
    return g, len(drug_drug_mat), len(protein_protein_mat), ntriples
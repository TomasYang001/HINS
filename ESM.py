#set the edit_idx
import pandas as pd
from tqdm import tqdm
import esm
# import TextCNN2 as tc2

from sklearn.metrics import roc_auc_score, average_precision_score

# import process_encoder
import numpy as np
from sklearn.metrics import accuracy_score

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import random_split
import argparse
# from Dataprocess import pre_process

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# dataset = pd.read_csv("NAgdata/Rizvi.csv")
#
# HLA_name = dataset["HLA"]
# Antigen = dataset["Antigen"]


#针对于没有HLA sequence的数据---------------------------------------------------------
def ESM2(Antigen, HLA_name):
    HLA_seq_lib = {}
    HLA = []
    #
    hla_db_dir = './NAgdata/HLA_library'
    HLA_ABC=[hla_db_dir+'/A_prot.fasta',hla_db_dir+'/B_prot.fasta',hla_db_dir+'/C_prot.fasta']

    for one_class in HLA_ABC:
        prot=open(one_class)
        #pseudo_seq from netMHCpan:https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0000796; minor bug 33 aa are used for pseudo seq, the performance is still good
        #HLA sequences are not aligned before taking pseudo-seq. but the performance is still good. will consider doing alignment before taking pseudo sequences in order to improve the performance
        pseudo_seq_pos = [7,9,24,45,59,62,63,66,67,79,70,73,74,76,77,80,81,84,95,97,99,114,116,118,143,147,150,152,156,158,159,163,167,171]
        #write HLA sequences into a library
        #class I alles
        name=''
        sequence=''
        for line in prot:
            if len(name)!=0:
                if line.startswith('>HLA'):
                    pseudo=''
                    for i in range(0,33):
                        if len(sequence)>pseudo_seq_pos[i]:
                            pseudo=pseudo+sequence[pseudo_seq_pos[i]]
                    HLA_seq_lib[name]=pseudo
                    name=line.split(' ')[1]
                    sequence=''
                else:
                    sequence=sequence+line.strip()
            else:
                name=line.split(' ')[1]


    for each_HLA in tqdm(HLA_name):
        if each_HLA not in HLA_seq_lib.keys():
            if len([hla_allele for hla_allele in HLA_seq_lib.keys() if hla_allele.startswith(str(each_HLA))]) == 0:
                print('cannot find' + each_HLA)
            each_HLA = [hla_allele for hla_allele in HLA_seq_lib.keys() if hla_allele.startswith(str(each_HLA))][0]
        if each_HLA not in HLA_seq_lib.keys():
            print('Not proper HLA allele:' + each_HLA)
        HLA.append(HLA_seq_lib[each_HLA])

    # print(HLA)
    #--------------------------------------------------------------------------------------

    # Load ESM-2 model
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model=model.to(device)
    batch_converter = alphabet.get_batch_converter()

    model.eval()  # disables dropout for deterministic results

    #Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)

    # print(2)
    Antigen_list = []
    HLA_list=[]
    for i in range(len(Antigen)):
        Antigen_ESM2_style=("protein"+str(i),Antigen[i].upper())
        HLA_ESM2_style=("protein"+str(i),HLA[i])
        Antigen_list.append(Antigen_ESM2_style)
        HLA_list.append(HLA_ESM2_style)


    Antigen_labels, Antigen_strs, Antigen_tokens = batch_converter(Antigen_list)
    HLA_labels, HLA_str, HLA_tokens = batch_converter(HLA_list)


    batch_size = 50
    num_batches = len(Antigen_labels) // batch_size + (len(HLA_list) % batch_size > 0)

    # result_attentions = (torch.Tensor(10, 33, 20, 27, 27))
    # result_representations=(torch.Tensor(10, 27,1280))
    model=model.to(device)
    print("Start ESM2 embedding...")

    Antigen_result=torch.zeros((len(Antigen_list), 17, 1280))
    HLA_result = torch.zeros((len(HLA_list), 35, 1280))


    for i in tqdm(range(num_batches)):
        start_index = i * batch_size
        if i==num_batches:
            end_index=len(Antigen_list)
        else:
            end_index = (i + 1) * batch_size
        Antigen_data = Antigen_tokens[start_index:end_index].to(device)
        HLA_data = HLA_tokens[start_index:end_index].to(device)
        with torch.no_grad():

            Antigen_embedding = model(Antigen_data, repr_layers=[33], return_contacts=True)
            HLA_embedding = model(HLA_data, repr_layers=[33], return_contacts=True)

            Antigen_result[start_index:end_index]=Antigen_embedding["representations"][33]
            HLA_result[start_index:end_index] = HLA_embedding["representations"][33]
    return Antigen_result, HLA_result

# Antigen_result, HLA_result = ESM2(Antigen, HLA_name)
#
# torch.save(Antigen_result,'V1_ESM2_Antigen.pt')
# torch.save(HLA_result,'V1_ESM2_HLA.pt')
import glob
import json
import os
from collections import defaultdict
import esm
import numpy as np
import pandas as pd
import torch
from nfold_Test import TestBAN
from Dataprocess import antigenMap, HLAMap, peptide_encode_HLA, hla_encode, pre_process
from ESM import ESM2
from tqdm import tqdm
import sys
import time

start = time.time()

paired = 'F'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Pat = sys.argv[1]

NAg = pd.read_csv(sys.argv[2])

NAg = NAg.dropna()
NAg = NAg.reset_index(drop=True)


def config():
    config = {}
    config['emb_size'] = 128
    config['dropout_rate'] = 0.1

    # DenseNet
    config['scale_down_ratio'] = 0.25
    config['growth_rate'] = 20
    config['transition_rate'] = 0.5
    config['num_dense_blocks'] = 4
    config['kernal_dense_size'] = 3

    # Encoder
    config['intermediate_size'] = 1536
    config['num_attention_heads'] = 8
    config['attention_probs_dropout_prob'] = 0.1
    config['hidden_dropout_prob'] = 0.1
    return config


class EpitopeDistance(object):
    """Base class for epitope crossreactivity.

    Model:
        dist({a_i}, {b_i}) = \sum_i d_i M_ab(a_i, b_i)

    Attributes
    ----------
    amino_acids : str
        Allowed amino acids in specified order.

    amino_acid_dict : dict
        Dictionary of amino acids and corresponding indicies

    d_i : ndarray
        Position scaling array d_i.
        d_i.shape == (9,)

    M_ab : ndarray
        Amino acid substitution matrix. Indexed by the order of amino_acids.
        M_ab.shape == (20, 20)

    """

    def __init__(self, model_file=os.path.join(os.path.dirname(__file__), 'NAgdata',
                                               'epitope_distance_model_parameters.json'),
                 amino_acids='ACDEFGHIKLMNPQRSTVWY'):
        """Initialize class and compute M_ab."""

        self.amino_acids = amino_acids
        # self.amino_acid_dict = {aa: i for i, aa in enumerate(self.amino_acids)}
        self.amino_acid_dict = {}
        for i, aa in enumerate(self.amino_acids):
            self.amino_acid_dict[aa.upper()] = i
            self.amino_acid_dict[aa.lower()] = i

        self.set_model(model_file)

    def set_model(self, model_file):
        """Load model and format substitution matrix M_ab."""
        with open(model_file, 'r') as modelf:
            c_model = json.load(modelf)
        self.d_i = c_model['d_i']
        self.M_ab_dict = c_model['M_ab']
        M_ab = np.zeros((len(self.amino_acids), len(self.amino_acids)))
        for i, aaA in enumerate(self.amino_acids):
            for j, aaB in enumerate(self.amino_acids):
                M_ab[i, j] = self.M_ab_dict[aaA + '->' + aaB]
        self.M_ab = M_ab

    def epitope_dist(self, epiA, epiB):
        """Compute the model difference between the 9-mers epiA and epiB.

        Ignores capitalization.

        Model:
            dist({a_i}, {b_i}) = \sum_i d_i M_ab(a_i, b_i)
        """

        return sum([self.d_i[i] * self.M_ab[self.amino_acid_dict[epiA[i]], self.amino_acid_dict[epiB[i]]] for i in
                    range(len(epiB))])


# immuno_score = []
count = 0
INB = []

epidist = EpitopeDistance()
w = 0.22402192838740312
Cf = config()
model = TestBAN(**Cf).to(device)

model.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu')))

model.eval()

clinical_record = []
patient = pd.DataFrame()

patient_Antigen = []
patient_allele = []

# WT_peptide_array = np.ndarray()
# HLA_array = np.ndarray()



patient_Antigen, patient_allele = pre_process(NAg)
# patient_Antigen.append("ARNLVPMVATVQGQN")
# patient_allele.append("A*02:01")

WT_peptide_B_array = antigenMap(patient_Antigen, 17, "BLOSUM62")
HLA_B_array = HLAMap(patient_allele, "BLOSUM62")
WT_peptide_B_array = torch.FloatTensor(WT_peptide_B_array)
HLA_B_array = torch.FloatTensor(HLA_B_array)

# WT_peptide_E_array, HLA_E_array = ESM2(patient_Antigen, patient_allele)

# WT_peptide_array = torch.cat((WT_peptide_B_array, WT_peptide_E_array), 2)
# HLA_array = torch.cat((HLA_B_array, HLA_E_array), 2)
# print("Size:", HLA_array.size())

count = 0
for neo in range(len(NAg) - 1):
    HLA_ = HLA_B_array[neo]
    HLA_ = HLA_.reshape(1, HLA_.shape[0], HLA_.shape[1])

    HLA_ = torch.Tensor(HLA_)
    Antigen_ = WT_peptide_B_array[neo]
    Antigen_ = Antigen_.reshape(1, Antigen_.shape[0], Antigen_.shape[1])
    Antigen_ = torch.Tensor(Antigen_)
    R = model(HLA_, Antigen_)
    A = np.log(int(NAg["WT_affinity"][neo]) / int(NAg["MT_affinity"][neo]))

    if len(NAg["WT_peptide"][neo]) >= 9:
        try:
            C = epidist.epitope_dist(NAg["Antigen"][neo], NAg["WT_peptide"][neo])
        except:
            C = 0
        quality = (w * C + (1 - w) * A) * abs(R)

    else:
        quality = 0

    # immuno_score.append(R)
    if quality > 1:
        count += 1

LgINB = np.log10(count + 1)
clinical_record.append([Pat, count, LgINB])

HED = 8.23
HINS = HED * LgINB

end = time.time()
Time = end - start
print("Patient:", Pat, "INB: ", count, "log(INB+1): ", LgINB, "HINS: ", HINS, "Time cost: ", Time)



# Prediction and multi-omics analysis of immune checkpoint blockade response by deep learning model with neoantigen quality

Immune checkpoint blockade (ICB) therapy response prediction provides remarkable genomic and transcriptomic gains and has successfully understood the molecular mechanisms underlying various cancers. However, only a subset of patients with advanced tumors currently benefit from ICB therapies, which at times incur considerable side effects and costs. Developing predictive tools for ICB response has remained a serious challenge because of the complexity of the immune response and the shortage of large cohorts of ICB-treated patients that include both ‘omics’ and response data. Here, based on a pooled pan-cancer genomic dataset of 919 patients treated with anti-PD-1 or anti-CTLA-4, we constructed human leukocyte antigen class I (HLA- I)-based immunogenic neoantigen score (HINS). HINS is a predictor of ICB response using deep attention networks and HLA evolutionary divergence to integrate the factors associated with immune activation and evasion. It yields an overall accuracy of AUC=0.853, outperforming existing predictors and capturing more than 75% true responders. Experimental analysis indicated that patients with higher HINS were more likely to undergo survival benefits following ICB therapy. Our results highlight the transcriptional and genomic correlations between HINS-identified molecule features and ICB response, such as immunoresponsive gene enrich pathways, favorable and unfavorable genomic subgroups, and hotspot somatic events in diver genes. This study presents an interpretable, accurate deep-learning method using meaningful predictive features to predict the response to immune checkpoint blockade, providing genomic insights into the complexity of determinants underlying immunotherapy.

## Graphic Abstract
<p align="center">
<img src="https://github.com/TomasYang001/HINS/blob/main/Graphic_Abstract.png" align="middle" height="80%" width="80%" />
</p>

## The environment of HINS
```
python==3.10
torch=2.2.1
numpy==1.19
fair-esm
```

## Installation Guide
Clone this Github repo and set up a new conda environment. It normally takes about 10 minutes to install on a normal desktop computer.
```
# create a new conda environment
$ conda create --name HINS python=3.10
$ conda activate HINS

# install requried python dependencies
$ conda install pytorch=2.2.1 torchvision torchaudio cudatoolkit=12.1 -c pytorch
$ pip install fair-esm
$ pip install -U scikit-learn
$ pip install yacs
$ pip install prettytable

# clone the source code of HINS
$ git clone https://github.com/TomasYang001/HINS.git
```

## Dataset description
In this paper, neoantigen data for model training are downloaded from IEDB (https://www.iedb.org/). 



## Run HINS
(1)Data embedding:
Please use the file Dataprocess.py to perform the embedding of Blosum62、Blosum50 or One-hot
Please use the file ESM.py to perform the embedding of ESM2-650

(2)5-fold CV for deep attention networks(DANs):
Blosum62 only
```sh
python nfold_Blosum.py
```

Blosum62 plus EMS2:
Blosum62 only
```sh
bash nfold.sh
```

(3)Calculating neoanitgen quality with the file Compute_quality.py

(4)Using HINS to predict the response to ICB therapy with the file Prediction.py


# Acknowledgments
The authors sincerely hope to receive any suggestions from you!

Of note, each embedding data, such as V1_B_train_label.npy, V1_B_train_antigen.npy of HINS in our study is too large to upload. 
If readers want to further study the ICB response prediction according to our study, please contact us at once. 
Email: YJ2197224605@163.com

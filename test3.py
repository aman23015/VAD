import os
import csv
import math
import torch
import numpy
import random 
import librosa
import argparse
import soundfile
import torchaudio
import torch.nn as nn
from model import*
from func import*
import torch.optim as optim
import torch.nn.functional as F
import scipy.signal as scisignal
from typing import List,Dict,Tuple
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


# P = argparse.ArgumentParser()
# P.add_argument("gpu",type=int,default=0)
# P.add_argument("epochs",type=int,default=64)
# P.add_argument("batch_size",type=int,default=64)
# P.add_argument("optimizer",type=str,default="adam")
# P.add_argument("experiment_name",type=str,default="vad")
# A = P.parse_args() 

##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++##
#parameters & paths
AVA_data = "/home/hiddencloud/AMAN_MT23015/DATA/AVA/AVA"
AVA_csv = "/home/hiddencloud/AMAN_MT23015/DATA/AVA_CSV"

exclude = "ava_speech_labels_v1.csv"
LOGPATH ="/home/hiddencloud/AMAN_MT23015"

# cuda = A.gpu



## Dataset
rttm_dict = {}
for i in sorted(os.listdir(AVA_csv)):
    csv_path = os.path.join(AVA_csv,i)
    with open(csv_path,'r',newline='') as csvfile:
        reader = csv.reader(csvfile)
        first_row = next(reader)
        last_row = []
        for j in reader:
            if j :
                last_row=j
        var = i.split('.')[0]
        rttm_dict[var] = (first_row[1],last_row[2])

print(rttm_dict)

def metadata():
    all_audios = sorted([os.path.join(AVA_data,i) for i in os .listdir(AVA_data) if i not in exclude])
    all_csv = sorted([os.path.join(AVA_csv,i) for i in os.listdir(AVA_csv)])
    dev_audios, test_audios, dev_csv, test_csv = train_test_split(all_audios, all_csv, test_size=0.2, random_state=42)
    return dev_audios,dev_csv,test_audios,test_csv


DevAudios , DevRTTMS , TestAudios , TestRTTMS = metadata()
# print("TestAudios : ")
# for i in TestAudios:
#     print(i)

# print("TestRTTMS : ")
# for i in TestRTTMS:
#     print(i)    


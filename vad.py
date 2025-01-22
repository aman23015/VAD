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


P = argparse.ArgumentParser()
P.add_argument("gpu",type=int,default=0)
P.add_argument("epochs",type=int,default=64)
P.add_argument("batch_size",type=int,default=64)
P.add_argument("optimizer",type=str,default="adam")
P.add_argument("experiment_name",type=str,default="vad")
A = P.parse_args() 

##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++##
#parameters & paths
AVA_data = "/home/hiddencloud/AMAN_MT23015/DATA/AVA/AVA"
AVA_csv = "/home/hiddencloud/AMAN_MT23015/DATA/AVA_CSV"

exclude = "ava_speech_labels_v1.csv"
LOGPATH ="/home/hiddencloud/AMAN_MT23015"

cuda = A.gpu

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

def metadata():
    all_audios = sorted([os.path.join(AVA_data,i) for i in os .listdir(AVA_data) if i not in exclude])
    all_csv = sorted([os.path.join(AVA_csv,i) for i in os.listdir(AVA_csv)])
    dev_audios, test_audios, dev_csv, test_csv = train_test_split(all_audios, all_csv, test_size=0.2, random_state=42)
    return dev_audios,dev_csv,test_audios,test_csv


DevAudios , DevRTTMS , TestAudios , TestRTTMS = metadata()


class ChunkedData(torch.utils.data.IterableDataset):
    def __init__(self,audio_path,rttm_path,rttm_dict):
        super().__init__()

        var = audio_path.split('/')
        var = var[-1].split('.')[0]
        self.audio_path = audio_path
        self.start_time,self.end_time = rttm_dict[var]
        self.audio,self.sr = self.load_audio_segment()
        self.labels = rttm_reader(style="AvaSpeech",path=rttm_path) if rttm_path != None else None
        self.window = 0.5
        self.overlap = 0.25

    def load_audio_segment(self):
        waveform,sample_rate = torchaudio.load(self.audio_path)
        start_sample = int(900*sample_rate)
        end_sample = int(1800*sample_rate)
        segment = waveform[:,start_sample:end_sample]
        return segment,sample_rate

    def rttm_to_labels(self):
        speaker_map = {"NO_SPEECH ":0,"CLEAN_SPEECH ":1,"SPEECH_WITH_MUSIC ":1,"SPEECH_WITH_NOISE ":1}

        if self.labels == None:
            return None
        
        speaker = []
        for idx,data in enumerate(self.labels):
            start,end = list(data[idx].values())[0][0],list(data[idx].values())[0][1]
            start = start - 900
            end = end - 900           
            if idx == 0:
                if start != 0:
                    speaker.append((speaker_map[list(data[idx].keys())[0]],0*self.sr,int(start*self.sr)))
                else :
                    speaker.append((speaker_map[list(data[idx].keys())[0]],0*self.sr,int(self.sr*end)))
            else: speaker.append((speaker_map[list(data[idx].keys())[0]],int(start*self.sr),int(end*self.sr)))
            
        # self.indexes = [() for k in range(0, self.audio.shape[-1]-int(self.sr*self.window)+1,int(self.sr*self.overlap))] 
        self.indexes = []
        step = int(self.sr * self.overlap)
        for k in range(0, self.audio.shape[-1] - int(self.sr * self.window) + 1, step):
            start_index = k
            end_index = k + int(self.sr * self.window)
            self.indexes.append((start_index, end_index))

        labels = []
        for idx in self.indexes:
            s = []
            s_ = ()
            for spk in speaker:
                if idx[0] <= spk[2]:
                    r1 , r2 = range(idx[0],idx[1]),range(spk[1],spk[2])
                    var = len(range(max(r1[0],r2[0]),min(r1[-1],r2[-1])+1))
                    if var != 0:
                        s_ = spk
                        s.append(spk[0])
                    else: pass
                else :pass   
            if len(s) == 1:
                labels.append(s[0])
            elif len(s) == 0:
                pass 
            elif len(s)>1: 
                labels.append(int(sum(s)/len(s)))        

        return torch.tensor(labels)  

    def __iter__(self):
        A = self.audio[0].unfold(0,size=int(self.sr*self.window),step=int(self.sr*self.overlap))
        L = self.rttm_to_labels()
        for i in range(len(self.indexes)):
            var = A[i].unsqueeze(0)
            yield var,L[i]


    
##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++##
##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++##

# Marblenet model 

MarbleNet_model = Marble_Net()
MarbleNet_model.train()


#Criterion
criterion = nn.CrossEntropyLoss()

##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++##
##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++##

#optimizer
if A.optimizer == "adam":
    optimizer = torch.optim.Adam(params=list(MarbleNet_model.parameters()),
                                 lr=0.0003,
                                 weight_decay=1e-3)
elif A.optimizer == "sgd":
    optimizer = torch.optim.SGD(params=list(MarbleNet_model.parameters()),
                                lr=0.0003,
                                momentum=0.9,                                
                                weight_decay=1e-3)    
    
##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++##
##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++##    

class VAD:
    def __init__(self,
                    vad_architecture,
                    criterion,
                    optimizer)->None:
        self.vad_net = vad_architecture
        self.criterion = criterion
        self.optimizer = optimizer

        self.logger = ChalBoard(exp_name = A.experiment_name,path=LOGPATH)
        self.device = torch.device(f"cuda:{cuda}") if cuda in range(0,torch.cuda.device_count()) else torch.device("cpu")

        self.vad_net.to(self.device)

    def audio_dataloader(self,audio_path,rttm_path):
        return DataLoader(dataset = ChunkedData(audio_path = audio_path,
                                                rttm_path = rttm_path,rttm_dict=rttm_dict),
                                                batch_size = A.batch_size,
                                                drop_last = False,
                                                num_workers = 2)

    def minibatch_process(self,batch,type):
        if type == "train":
            optimizer.zero_grad()

            data,label = batch
            data,label = data.to(self.device),label.to(self.device)
            predcs = self.vad_net(data)
            print(label)
            print(predcs)
            input("wait")
            loss = self.criterion(predcs,label)
            loss.backward()
            optimizer.step()
            accuracy = label.eq(F.softmax(predcs,dim=1).argmax(1)).sum().div(A.batch_size).mul(100)
            return loss.item(),accuracy.item()
        elif type == "eval" :
            data,label = batch
            data,label = data.to(self.device),label.to(self.device)
            predcs = self.vad_net(data)
            accuracy = label.eq(F.softmax(predcs,dim=1).argmax(1)).sum().div(A.batch_size).mul(100)
            return accuracy.item()
    
    def train(self):
        torch.cuda.empty_cache()
        best_acc = [70.00]
        for epoch in range(A.epochs):
            vad_loss ,vad_acc = 0 , 0

            for d_idx in range(len(DevAudios)):
                aud_loss , aud_acc, mb_idx = 0,0,0
                for minibatch in self.audio_dataloader(audio_path=DevAudios[d_idx],
                                                        rttm_path=DevRTTMS[d_idx]):
                    mb_idx += 1
                    loss,accuracy = self.minibatch_process(batch=minibatch,type="train")
                    aud_loss += loss 
                    aud_acc  += accuracy   
                print(
                    "Epoch: {}, Audio: {},[Loss: {}, Accuracy: {}]".format(
                        epoch,
                        d_idx,
                        round(aud_loss/mb_idx,4),
                        round(aud_acc/mb_idx,4)
                    ),
                    end = "\n"
                )  
                vad_loss += aud_loss/mb_idx
                vad_acc += aud_acc/mb_idx  

            details = "Epoch: {},[Loss: {},Accuracy: {},]".format(
                epoch,
                round(vad_loss/len(DevAudios),4),
                round(vad_acc/len(DevAudios),4)
            )    
            print(details)
            self.logger.scribe(details)

            if(epoch%10==0 and epoch!=0):
                with torch.no_grad():
                    print("Test set : ")
                    test_vad_acc = 0
                    for d_idx in range(len(TestAudios)):
                        test_aud_acc,mb_idx = 0,0
                        for minibatch in self.audio_dataloader(audio_path=TestAudios[d_idx],
                                                            rttm_path=TestRTTMS[d_idx]):
                            mb_idx += 1
                            accuracy_ = self.minibatch_process(batch=minibatch,type="eval")
                            test_aud_acc += accuracy_
                        print(
                            "Epoch: {}, Audio: {}, Accuracy: {}".format(
                                epoch,
                                d_idx,
                                round(test_aud_acc/mb_idx,4)
                            ),
                            end = "\n"
                        )   
                        test_vad_acc += test_aud_acc/mb_idx
                    details = "Epoch: {},Accuracy: {}".format(
                        epoch,
                        round(test_vad_acc/len(DevAudios),4)
                    )    
                    print(details)
                    self.logger.scribe(details)    




            if round(vad_acc/len(DevAudios),4) > best_acc[0]:
                torch.save(obj={"vad": self.vad_net.state_dict()},
                        f=os.path.join(self.logger.exp_path,f"checkpoint.pth"))
                best_acc[0] = round(vad_acc/len(DevAudios),4)
            else:pass    

                    

if __name__ == "__main__":
    Trainer = VAD(vad_architecture=MarbleNet_model,
                  criterion=criterion,
                  optimizer=optimizer)
    Trainer.train()                   

                                                        

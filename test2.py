import os 
import math
import torch 
import numpy
import random
import librosa
import soundfile
import torchaudio
import torch.nn.functional as F
import scipy.signal as scisignal
from typing import List,Dict,Tuple

AVA_data = "/home/hiddencloud/AMAN_MT23015/DATA/AVA/AVA"
AVA_csv = "/home/hiddencloud/AMAN_MT23015/DATA/AVA_CSV"

def rttm_reader(style="displace24", path:str=None):
    if style == "displace24":
        with open(path, "r") as F: lines = F.readlines(); F.close()
        lines = ["-".join([j for j in i.replace("<NA>", "").replace("\n", "").split(" ")[3:] if j != " "]).replace("--", "").replace("---", "-") for i in lines]
        labels = [
            {idx: {i.split("-")[-1]: [float(i) for i in i.split("-")[:2]] }} for idx, i in enumerate(lines)
            ]
        return labels
    elif style == "AvaSpeech" : 
        with open(path,"r") as F: lines = F.readlines(); F.close()
        lines = ["-".join(j for j in i.replace("\n"," ").split(",")[1:])for i in lines]
        labels = [
            {idx: {i.split("-")[-1]: [float(i) for i in i.split("-")[:2]] }} for idx, i in enumerate(lines)
            ]
        return labels

class Audio:
    def __init__(self)->None:
        pass

    def load(self,
             path:str,
             audio_duration = 2,
             sample_rate:int = 16000,
             backend:str = "torchaudio",
             audio_normalization:bool = True,
             audio_concat_strategy : str="flip_n_join"):
        
        if backend not in ["torchaudio","librosa","soundfile"]:
            raise Exception(f"Only implemented for (torchaudio,librosa,soundfile)")
        if audio_concat_strategy not in ["flip_n_join","repeat"]:
            raise Exception(f"Only implemented for (random_concat,flip_n_join,repeat)")
        
        if backend == "torchaudio":
            audio,sr = torchaudio.load(path)
        if backend == "librosa":
            audio,sr = librosa.load(path)
            audio = torch.tensor(audio,dtype=torch.float32).unsqueeze(0)
        if backend == "soundfile":
            audio,sr = soundfile.read(path)
            audio = torch.tensor(audio,dtype = torch.float32).unsqueeze(0)

        max_frames = audio_duration*sample_rate

        if sample_rate!=sr:
            resampler = torchaudio.transforms.Resample(sr,sample_rate,dtype=audio.dtype)
            audio = resampler(audio)
        else : pass

        if audio_duration == "full":
            if audio_normalization:
                audio = torch.nn.functional.normalize(audio)
            else : pass
            return audio,sample_rate
        
        if audio.shape[1] < max_frames:
            if audio_concat_strategy == "flip_n_join":
                audio = torch.cat([audio,audio.flip((1,))]*int(max_frames/audio.shape[1]),dim=1)[0][:max_frames]
            if audio_concat_strategy == "repeat":
                audio = torch.tile(audio,(math.ceil(max_frames/audio.shape[1])))[0][:max_frames]    
        else :
            start = random.randint(0,audio.shape[1]-max_frames + 1)
            audio = audio[0][start:start+max_frames]

AVA_data = "/home/hiddencloud/AMAN_MT23015/DATA/AVA/AVA"
AVA_csv = "/home/hiddencloud/AMAN_MT23015/DATA/AVA_CSV"

exclude = "ava_speech_labels_v1.csv"
def metadata():
    dev_audios = sorted([os.path.join(AVA_data,i) for i in os .listdir(AVA_data) if i not in exclude])
    csv_audios = sorted([os.path.join(AVA_csv,i) for i in os.listdir(AVA_csv)])
    return dev_audios,csv_audios

DevAudios , DevRTTMS = metadata()
# print("number of audios and rttms")
# print(len(DevAudios))
# print(len(DevRTTMS))
# input("wait")   


class ChunkedData(torch.utils.data.IterableDataset):

    def __init__(self,audio_path,rttm_path):
        super().__init__()
        self.audio_path = audio_path
        self.audio,self.sr = self.load_audio_segment(start = 900,end = 1800)
        self.labels = rttm_reader(style="AvaSpeech",path=rttm_path) if rttm_path != None else None
        self.window = 1.5
        self.overlap = 0.5

    def load_audio_segment(self,start,end):
        waveform,sample_rate = torchaudio.load(self.audio_path)
        start_sample = int(start*sample_rate)
        end_sample = int(end*sample_rate)
        segment = waveform[:,start_sample:end_sample]
        return segment,sample_rate
    
    def rttm_to_labels(self):
        speaker_map = {"NO_SPEECH ":0,"CLEAN_SPEECH ":1,"SPEECH_WITH_MUSIC ":1,"SPEECH_WITH_NOISE ":1}

        if self.labels == None:
            return None
        speaker = []
        print(self.labels)
        for idx,data in enumerate(self.labels):
            start,end = list(data[idx].values())[0][0],list(data[idx].values())[0][1]
            start = start - 900
            end = end - 900
            if idx == 0:
                if start != 0:
                    speaker.append((speaker_map[list(data[idx].keys())[0]],0*self.sr,int(start*self.sr)))
                else :
                    speaker.append((speaker_map[list(data[idx].keys())[0]],0*self.sr,int(self.sr*end)))
            else: speaker.append((speaker_map[list(data[idx].keys())[0]],int(start*self.sr),int((end)*self.sr)))
        
        print(speaker)
        # self.indexes = [() for k in range(0, self.audio.shape[-1]-int(self.sr*self.window)+1,int(self.sr*self.overlap))] 
        # print("self.indexes : ",self.audio.shape[-1],"  ",int(self.sr*self.window)+1," ",int(self.sr*self.overlap)) 
        self.indexes = []
        step = int(self.sr * self.overlap)
        print(step)
        print(self.audio.shape[-1])
        print(int(self.sr * self.window))
        print(self.audio.shape[-1] - int(self.sr * self.window) + 1)
        input("wait")
        for k in range(0, self.audio.shape[-1] - int(self.sr * self.window) + 1, step):
            start_index = k
            end_index = k + int(self.sr * self.window)
            self.indexes.append((start_index, end_index))
            # print("start index ",start_index)
            # print("end index ", end_index)
            # print("self.indexes ", self.indexes)
            # input("wait")
        # print("length of self.indexes : ",len(self.indexes))
        # print("length of speaker : ",len(speaker))
        # print(self.indexes)
        # input("wait")
        labels = []
        for idx in self.indexes:
            s = []
            for spk in speaker:
                # print("idx : ",idx)
                # print("spk : ",spk)
                if idx[0] <= spk[2]:
                    r1 , r2 = range(idx[0],idx[1]),range(spk[1],spk[2])
                    var = len(range(max(r1[0],r2[0]),min(r1[-1],r2[-1])+1))
                    if var != 0:
                        # print(spk)
                        s_ = spk
                        s.append(spk[0])
                    else: pass
                else :pass
                # input("wait")
            # print("s list : ",s)    
            if len(s) == 1:
                labels.append(s[0])
            elif len(s) == 0:
                labels.append(2)  
            elif len(s)>1: 
                labels.append(int(sum(s)/len(s)))        
        # print("length of labels : ",len(labels))    
        # input("wait")
        # print(labels)

        return torch.tensor(labels)  

    def __iter__(self):
        A = self.audio[0].unfold(0,size=int(self.sr*self.window),step=int(self.sr*self.overlap))
        L = self.rttm_to_labels()
        for i in range(len(self.indexes)):
            print(A[i].shape)
            var = A[i].unsqueeze(0)
            print(var.shape)
            yield A[i],L[i]
              


        
                        
audio_path = "/home/hiddencloud/AMAN_MT23015/DATA/AVA/AVA/-5KQ66BBWC4.mp4"
rttm_path = "/home/hiddencloud/AMAN_MT23015/DATA/AVA_CSV/-5KQ66BBWC4.txt"

new_audio_path = "/home/hiddencloud/AMAN_MT23015/DATA/DISPLACE/2024/Displace2024_dev_audio_supervised/AUDIO_supervised/Track1_SD_Track2_LD/B007.wav"
new_rttm_path = "/home/hiddencloud/AMAN_MT23015/DATA/DISPLACE/2024/Displace2024_dev_labels_supervised/Labels/Track1_SD/B007_SPEAKER.rttm"

obj = ChunkedData(audio_path,rttm_path)
obj.rttm_to_labels()

# for i in DevAudios:
#     obj = ChunkedData(i,rttm_path)
#     print(obj)
#     print(obj.shape)
#     input("wait")

# for i in obj :
#     audio,label = i
#     print("loop")
#     print(audio)
#     print(audio.shape)
#     print(label)
#     input("wait")


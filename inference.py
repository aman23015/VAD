from model import*
from func import*
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# DISPLACE dataset
audio_path = "/home/hiddencloud/AMAN_MT23015/DATA/DISPLACE/2024/Displace2024_dev_audio_supervised/AUDIO_supervised/Track1_SD_Track2_LD/B007.wav"
rttm_path = "/home/hiddencloud/AMAN_MT23015/DATA/DISPLACE/2024/Displace2024_dev_labels_supervised/Labels/Track1_SD/B007_SPEAKER.rttm"

P = argparse.ArgumentParser()
P.add_argument("gpu",type=int,default=0)
# P.add_argument("epochs",type=int,default=64)
# P.add_argument("batch_size",type=int,default=64)
P.add_argument("experiment_name",type=str,default="vad")
A = P.parse_args() 

#AVA dataset
# audio_path = "/home/hiddencloud/AMAN_MT23015/DATA/AVA/AVA/-5KQ66BBWC4.mp4"
# rttm_path = "/home/hiddencloud/AMAN_MT23015/DATA/AVA_CSV/-5KQ66BBWC4.txt"
LOGPATH ="/home/hiddencloud/AMAN_MT23015"
cuda = A.gpu

class ChunkedData(torch.utils.data.IterableDataset):
    def __init__(self,audio_path,rttm_path,):
        super().__init__()

        var = audio_path.split('/')
        var = var[-1].split('.')[0]
        self.audio_path = audio_path
        # self.start_time,self.end_time = rttm_dict[var]
        self.audio,self.sr = self.load_audio_segment()
        self.labels = rttm_reader(style="displace24",path=rttm_path) if rttm_path != None else None
        self.window = 0.5
        self.overlap = 0.25

    def load_audio_segment(self):
        waveform,sample_rate = torchaudio.load(self.audio_path)
        start_sample = int(900*sample_rate)
        end_sample = int(1800*sample_rate)
        segment = waveform[:,start_sample:end_sample]
        return segment,sample_rate

    def rttm_to_labels(self):
        #AVA
        # speaker_map = {"NO_SPEECH ":0,"CLEAN_SPEECH ":1,"SPEECH_WITH_MUSIC ":1,"SPEECH_WITH_NOISE ":1} 

        #DISPLACE
        speaker_map ={"S1":0, "S2":1, "S3":2, "S4":3, "S5":4, "NA":5}

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

# Marblenet model
vad_model = Marble_Net()
vad_model.load_state_dict(
    state_dict=torch.load("/home/hiddencloud/AMAN_MT23015/chkpts/vad_2/checkpoint.pth")["vad"]
)


class Eval:
    def __init__(
            self,vad_architecture,audio_path,rttm_path
    )->None:
        self.vad_net = vad_architecture
        self.audio_path = audio_path
        self.rttm_path = rttm_path
        self.logger = ChalBoard(exp_name = A.experiment_name,path=LOGPATH)
        self.device = torch.device(f"cuda:{cuda}") if cuda in range(0,torch.cuda.device_count()) else torch.device("cpu")

        self.vad_net.to(self.device)

    def audio_dataloader(self,audio_path,rttm_path):
        return DataLoader(dataset = ChunkedData(audio_path = audio_path,
                                                rttm_path = rttm_path),
                                                batch_size = 1,
                                                drop_last = False,
                                                num_workers = 2)    
    

    def minibatch_process(self,batch):
            data,label = batch
            data,label = data.to(self.device),label.to(self.device)
            predcs = self.vad_net(data)
            accuracy = label.eq(F.softmax(predcs,dim=1).argmax(1)).sum().div(1).mul(100)
            return accuracy.item()
    
    def eval(self):
        torch.cuda.empty_cache()
        aud_acc,mb_idx = 0,0
        for minibatch in self.audio_dataloader(audio_path=self.audio_path,
                                               rttm_path=self.rttm_path):
            mb_idx += 1
            accuracy = self.minibatch_process(batch = minibatch)
            aud_acc += accuracy
        print(
            "Accuracy: {}".format(
                round(aud_acc/mb_idx,4)
            ),
            end = "\n"
        )    
        

        
if __name__ == "__main__":
    Evaluation = Eval(vad_architecture=vad_model,
                      audio_path=audio_path,
                      rttm_path=rttm_path)
    Evaluation.eval()


# def vad(VADModel,audio_path,rttm_path):
#     audio,sr = torchaudio.load(audio_path)
#     print("sample_rate : ",sr)
#     print("audio_shape : ",audio.shape)
#     input("wait")
#     SpeakerSegments = VADModel(audio)
#     print(len(SpeakerSegments))
#     for i in SpeakerSegments:
#         print(i)

# print("start")
# vad(vad_model,audio_path,rttm_path)
# print("done")        


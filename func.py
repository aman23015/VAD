
import os
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
    
class ChalBoard:
    def __init__(self,exp_name,path) ->None:

        self.exp_path = os.path.join(path,f"chkpts/{exp_name}")
        if not os.path.isdir(self.exp_path):
            os.makedirs(self.exp_path)  

        self.board_filepath = os.path.join(self.exp_path,"board.txt")
        with open(self.board_filepath,"a") as F: F.write("Experiment Details >> "+"\n"); F.close()

    def scribe(self,*args):
        with open(self.board_filepath,"a") as F:
            F.write(f">> >> " + ", ".join([str(i) for i in args]) + "\n")
            F.close        



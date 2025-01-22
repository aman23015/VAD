import os 
import csv
import torch 
import pickle
import pydub
from moviepy.editor import VideoFileClip, concatenate_videoclips

##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++##
#parameters & paths
AVA_data = "/home/hiddencloud/AMAN_MT23015/DATA/AVA/AVA"
AVA_csv = "/home/hiddencloud/AMAN_MT23015/DATA/AVA_CSV"

exclude = "ava_speech_labels_v1.csv"
LOGPATH ="/home/hiddencloud/AMAN_MT23015"

## Dataset
#/home/hiddencloud/AMAN_MT23015/DATA/AVA_CSV/-5KQ66BBWC4.txt


def metadata():
    dev_audios = sorted([os.path.join(AVA_data,i) for i in os .listdir(AVA_data) if i not in exclude])
    csv_audios = sorted([os.path.join(AVA_csv,i) for i in os.listdir(AVA_csv)])
    return dev_audios,csv_audios

DevAudios , DevRTTMS = metadata()

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



# def trim_audio(file_path, start,end,output_path):
#     print(file_path)
#     print(start)
#     print(end)
#     print(output_path)
#     input("wait")
#     clip = VideoFileClip(file_path)
#     audio_clip = clip.audio

#     if(start < 0 or start>=clip.duration):
#         print(f"Error: start time ({start}s) is outside video duration ({clip.duration}s).")
#         return
    
#     if end <= start or end > clip.duration:
#         print(f"Error: End time ({end}s) is outside vedio duration")
#         return
#     trimmed_audio = audio_clip.subclip(start,end)
#     trimmed_clip = VideoFileClip(clip.reader.rawduration,fps=clip.fps)
#     trimmed_clip = trimmed_clip.set_audio(trimmed_audio)
#     trimmed_clip.write_videofile(output_path)

# def trim_audio(file_path,start,end,output_path):
#     with pydub.AudioSegment.from_wav(file_path) as audio:
#         trimmed_audio = audio[int(start * 1000):int(end * 1000)]
#         trimmed_audio.export(output_path,format="mp4")
#         print("trimmed")


# new_path = "/home/hiddencloud/AMAN_MT23015/VAD/AVA"


# for _,i in enumerate(rttm_dict):
#     name = i+".mp4"
#     path = os.path.join(new_path,name)
#     # if not os.path.isdir(path):
#     #     os.makedirs(path)
#     start,end = rttm_dict[i][0],rttm_dict[i][1]
#     file_path = DevAudios[_]
#     trim_audio(file_path=file_path,
#                start=start,
#                end=end,
#                output_path=path)

# print("done")        
        


    


#ava_speech_labels_v1.csv
# data = {}
# with open(rttm_path, 'r', newline='') as csvfile:
#     reader = csv.reader(csvfile)
#     for i,row in enumerate(reader):
#         id = row[0]
#         if id not in data:
#             data[id] = [i+1,i+1]
#         else :
#             data[id][1] = i+1 

# print(data)
# print(len(data))

# ava_csv_folder = "AVA_CSV_1"
# data = {}
# demo=[]
# with open(rttm_path, 'r') as csvfile:
#     reader = csv.reader(csvfile)       
#     for row in reader:
#         id = row[0] 
#         var = id + ".mp4"
#         if var in os.listdir(data_path):
#             if id not in data:
#                 data[id] = []
#             data[id].append(row)

#     for id, rows in data.items():
#         output_file = os.path.join(ava_csv_folder, f"{id}")
#         with open(f"{output_file}.txt", 'w') as outfile:
#             writer = csv.writer(outfile)
#             writer.writerows(rows)


# 

# def save_dict_to_pickle(data, filename):
#   with open(filename, 'wb') as outfile:
#     pickle.dump(data, outfile)


# # save_dict_to_pickle(data, "data.pkl")
# with open(rttm_path,'r',newline='')as csvfile:
#     reader = csv.reader(csvfile)
#     with open("data.pkl", 'rb') as infile:  
#         data = pickle.load(infile)
#         for row in data:
#             i = data[row][0]
#             j = data[row][1]
#             print(i)
#             print(j)
#             input("wait")
#             for k in range(i-1,j+1):
#                 print(reader[k])
#             print("done")
#             input("wait")    

        
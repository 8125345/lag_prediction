import os
import glob
base_path = '/deepiano_data/zhaoliang/SC55_data/Alignment_data/split_320_npy'
song_ID = glob.glob(os.path.join(base_path, '*/*/*'))
for song in sorted(song_ID):
    # print(song)
    if os.path.isdir(song):
        npy_file_list = glob.glob(os.path.join(song, '*.npy'))
        if len(npy_file_list) == 0:
            print(f'{song}文件为空')

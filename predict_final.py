import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 推理时限定GPU

import glob
import json
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

batch_size = 1
lag_type = 'lag_32'


def parse_np_data(path):
    data = np.load(path)
    chunk_spec = data[:, : 229]
    onsets_label = data[:, 229:]
    return chunk_spec, onsets_label


def get_all_npy_data(song_folder_path):
    assert os.path.exists(song_folder_path)
    file_paths = glob.glob(os.path.join(song_folder_path, "*.npy"))

    def sort_fun(path):
        f_name = os.path.split(path)[-1]
        return int(os.path.splitext(f_name)[0])
    if file_paths:
        file_paths = sorted(file_paths, key=sort_fun)
    return file_paths


def model_predict(lag_type, wav_data, midi_data):
    print(f"模型类型:{lag_type}")
    if lag_type == "lag_160":
        model_path = "/data/projects/LabelModels/lag_prediction/train_output_models/cornet_20220926_3.h5"  # +-160
    elif lag_type == "lag_32":
        model_path = "/data/projects/LabelModels/lag_prediction/train_output_models/cornet_20220926_2.h5"  # +-32
    elif lag_type == "lag_8":
        model_path = "/data/projects/LabelModels/lag_prediction/train_output_models/cornet_20220928_0.h5"  # +-8
    else:
        raise Exception("类型错误")
    model = load_model(model_path, compile=False)

    feature0_arr = wav_data
    feature1_arr = midi_data

    rst = model.predict([feature0_arr, feature1_arr], batch_size=batch_size)
    return rst

def batch_predict(base_path):
    assert os.path.exists(base_path)
    equipment_ID_list = glob.glob(os.path.join(base_path, '*'))
    for equipment_ID in sorted(equipment_ID_list):
        if os.path.isfile(equipment_ID):
            continue
        song_ID_list = glob.glob(os.path.join(equipment_ID, '*/*'))
        for song_ID in sorted(song_ID_list):
            npy_file_paths = get_all_npy_data(song_ID)
            for npy_file in npy_file_paths:
                chunk_spec, onsets_label = parse_np_data(npy_file)
                result = model_predict(lag_type, chunk_spec, onsets_label)


def single_predict_for_test(audio_single):
    chunk_list = list()
    onset_list = list()
    npy_file_paths = get_all_npy_data(audio_single)
    for npy_file in npy_file_paths:
        chunk_spec, onsets_label = parse_np_data(npy_file)
        chunk_spec = chunk_spec.reshape(chunk_spec.shape[0], chunk_spec.shape[1], 1)
        onsets_label = onsets_label.reshape(onsets_label.shape[0], onsets_label.shape[1], 1)
        chunk_list.append(chunk_spec)
        onset_list.append(onsets_label)
    stack_chunk = np.array(chunk_list)  # 音频数据
    stack_onset = np.array(onset_list)  # midi数据

    result = model_predict(lag_type, stack_chunk, stack_onset)
    # print(result.shape)
    # print(result)
    return result


if __name__ == '__main__':
    base_path = '/deepiano_data/zhaoliang/SC55_data/Alignment_data/split_320_npy'
    # batch_predict(base_path)
    test_base_path = '/deepiano_data/zhaoliang/SC55_data/Alignment_data/split_320_npy/ipad6SC55录音版/total/'
    test_audio_list = [
    'xml_arachno_000',  #midi相对于wav的初始帧误差为:[5.752642]	误差均值为:[5.752642] 实际误差150ms
    'xml_arachno_001',  #midi相对于wav的初始帧误差为:[5.2685623]	误差均值为:[5.2685623] 实际误差150ms
    'xml_arachno_003',  #midi相对于wav的初始帧误差为:[5.8328]	误差均值为:[4.228967]  实际误差159ms
    'xml_arachno_016',  #midi相对于wav的初始帧误差为:[4.1683326]	误差均值为:[4.1683326] 实际误差155ms
    'xml_arachno_020',  #midi相对于wav的初始帧误差为:[7.5355844]	误差均值为:[6.8415875] 实际误差160ms
    'xml_arachno_050',  #midi相对于wav的初始帧误差为:[3.7279887]	误差均值为:[3.7279887] 实际误差150ms
    'xml_arachno_060',  #midi相对于wav的初始帧误差为:[4.941848]	误差均值为:[4.941848] 实际误差157ms
    'xml_arachno_080',  #midi相对于wav的初始帧误差为:[2.9952936]	误差均值为:[3.6221085] 实际误差157ms
    'xml_arachno_090',  #midi相对于wav的初始帧误差为:[5.8857493]	误差均值为:[5.4263244] 实际误差165ms
    'xml_arachno_100',  #midi相对于wav的初始帧误差为:[3.8128076]	误差均值为:[1.136018] 实际误差174ms
    'xml_arachno_110',  #midi相对于wav的初始帧误差为:[6.2852097]	误差均值为:[7.346863] 实际误差165ms
    'xml_arachno_120',  #midi相对于wav的初始帧误差为:[6.391454]	误差均值为:[6.7817974] 实际误差170ms
    'xml_arachno_130',  #midi相对于wav的初始帧误差为:[5.779112]	误差均值为:[5.1921444] 实际误差168ms
    ]
    for test_audio in test_audio_list:
        input_audio = test_base_path + test_audio
        assert os.path.exists(input_audio)
        result = single_predict_for_test(input_audio)
        print(f'文件{input_audio}midi相对于wav的初始帧误差为:{result[0]}\t误差均值为:{result.mean(axis=0)}')











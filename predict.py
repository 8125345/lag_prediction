import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 推理时限定GPU

import json
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np


def model_predict(lag_type):
    # 加载模型
    # 说明，lag后面的数字代表模型能接受的数据最大错位，例如lag_320代表输入模型的频谱和标签可以有+-160帧的错位
    # 对错位容忍越大的模型精度越低，以下是平均误差，单位：帧。（maestro验证集上的mae值）
    # lag_160: +-10; lag_32: +-2; lag_8: +-0.85
    # 可以使用级联推理逐步降低误差

    print(f"模型类型:{lag_type}")
    if lag_type == "lag_160":
        model_path = "/data/projects/LabelModels/lag_prediction/train_output_models/cornet_20220926_3.h5"  # +-160
        lag_list = [-160, -100, -50, -5, -1, 0, 1, 5, 50, 100, 160]
        # lag_list = [-32, -16, -10, -5, -1, 0, 1, 5, 10, 16, 32]
        # lag_list = [-8, -6, -3, -2, -1, 0, 1, 2, 3, 6, 8]

        # lag_list = [0, 0, 0, 0, 0, 0, 0]
    elif lag_type == "lag_32":
        model_path = "/data/projects/LabelModels/lag_prediction/train_output_models/cornet_20220926_2.h5"  # +-32
        lag_list = [-32, -16, -10, -5, -1, 0, 1, 5, 10, 16, 32]
        # lag_list = [-8, -6, -3, -2, -1, 0, 1, 2, 3, 6, 8]

    elif lag_type == "lag_8":
        model_path = "/data/projects/LabelModels/lag_prediction/train_output_models/cornet_20220928_0.h5"  # +-8
        lag_list = [-8, -6, -3, -2, -1, 0, 1, 2, 3, 6, 8]
        # lag_list = [-160, -100, -50, -5, -1, 0, 1, 5, 50, 100, 160]

    else:
        raise Exception("类型错误")
    model = load_model(model_path, compile=False)

    # 读取数据
    json_path = "/deepiano_data/zhaoliang/public_data/json_mix_public/maestro-v3.0.0_train.json"
    with open(json_path, "r") as f:
        data = json.load(f)

    # 抽几条数据
    cnt = 0
    feature0_list = list()
    feature1_list = list()

    label_list = list()

    limit = len(lag_list)
    mel_start_idx = 160  # 从640帧序列中开始截取的位置
    mel_len = 320  # 模型输入的mel频谱图长度

    for _, path in data.items():
        cnt += 1
        serialized = tf.io.read_file(path)
        data = tf.io.parse_tensor(serialized, tf.float32)
        feature = data[:, : 293120]
        midi = data[:, 293120:]
        feature = tf.reshape(feature, (640, 229, 2))
        feature = feature[:, :, 1]  # 获取mix
        midi = tf.reshape(midi, (640, 88))

        feature = feature.numpy()
        midi = midi.numpy()

        feature = feature[mel_start_idx: mel_start_idx + mel_len, :]
        lag_value = lag_list[cnt - 1]
        midi = midi[mel_start_idx + lag_value: mel_start_idx + mel_len + lag_value, :]

        # 使用中间的音频，测试+/-10帧错误预测结果
        feature0_list.append(feature)
        feature1_list.append(midi)
        label_list.append(lag_value)

        if cnt == limit:
            break

    feature0_arr = np.array(feature0_list) #音频数据
    feature1_arr = np.array(feature1_list) #midi数据

    label_arr = np.array(label_list) #错位帧数，label

    batch_size = 64  # 尽量吧GPU打满
    rst = model.predict([feature0_arr, feature1_arr], batch_size=batch_size)

    # print(rst.shape)
    # print(rst)
    # 实验结果
    data_num = rst.shape[0]
    for i in range(data_num):
        predict_v = rst[i][0]
        gt_v = label_arr[i]

        print(f"模型预测:{np.around(float(predict_v), 2)}\t标签:{gt_v}\t误差: {np.around(gt_v - predict_v, 2)}")
    print("=" * 100)


def run():
    for lag_type in ["lag_8", "lag_32", "lag_160"]:
        model_predict(lag_type)


if __name__ == '__main__':
    run()
# ====================================================================================================
# 模型预测:-6.88	标签:-8	误差: -1.12
# 模型预测:-5.75	标签:-6	误差: -0.25
# 模型预测:-3.34	标签:-3	误差: 0.34
# 模型预测:-1.81	标签:-2	误差: -0.19
# 模型预测:-1.62	标签:-1	误差: 0.62
# 模型预测:0.35	标签:0	误差: -0.35
# 模型预测:0.58	标签:1	误差: 0.42
# 模型预测:1.57	标签:2	误差: 0.43
# 模型预测:2.36	标签:3	误差: 0.64
# 模型预测:5.83	标签:6	误差: 0.17
# 模型预测:7.07	标签:8	误差: 0.93
# ====================================================================================================
# 模型类型:lag_32
# 模型预测:-30.67	标签:-32	误差: -1.33
# 模型预测:-15.4	标签:-16	误差: -0.6
# 模型预测:-11.15	标签:-10	误差: 1.15
# 模型预测:-4.7	标签:-5	误差: -0.3
# 模型预测:-1.56	标签:-1	误差: 0.56
# 模型预测:0.13	标签:0	误差: -0.13
# 模型预测:0.74	标签:1	误差: 0.26
# 模型预测:4.92	标签:5	误差: 0.08
# 模型预测:10.28	标签:10	误差: -0.28
# 模型预测:16.66	标签:16	误差: -0.66
# 模型预测:31.44	标签:32	误差: 0.56
# ====================================================================================================
# 模型类型:lag_160
# 模型预测:-155.74	标签:-160	误差: -4.26
# 模型预测:-99.32	标签:-100	误差: -0.68
# 模型预测:-48.87	标签:-50	误差: -1.13
# 模型预测:-3.36	标签:-5	误差: -1.64
# 模型预测:0.47	标签:-1	误差: -1.47
# 模型预测:1.74	标签:0	误差: -1.74
# 模型预测:1.04	标签:1	误差: -0.04
# 模型预测:5.76	标签:5	误差: -0.76
# 模型预测:50.95	标签:50	误差: -0.95
# 模型预测:101.82	标签:100	误差: -1.82
# 模型预测:158.74	标签:160	误差: 1.26
# ====================================================================================================

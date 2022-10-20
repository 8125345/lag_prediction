import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import imageio
import json
import tensorflow as tf

json_path = "/deepiano_data/yuxiaofei/work/data_0718/serialize_changpu_delay_metronome/ai-tagging_train.json"
with open(json_path, "r") as f:
    data = json.load(f)
# ".serialized"数据
test_index = 3  # 随机选择测试片段
serialized_path = data[str(test_index)]  # 序列化数据路径


def load_and_parse_data(path):
    """
    加载并解析数据
    :param path:
    :return:
    """

    serialized = tf.io.read_file(path)
    data = tf.io.parse_tensor(serialized, tf.float32)

    return data


data = load_and_parse_data(serialized_path)

feature = data[:, : 293120]
midi = data[:, 293120:]
feature = tf.reshape(feature, (640, 229, 2))  # 640, 229, 1
feature_max_value = tf.math.reduce_max(feature)
midi = tf.reshape(midi, (640, 88))  # 640 88
feature = feature[:, :, 1:]

imageio.imwrite("full_mel.png", feature / feature_max_value)
imageio.imwrite("full_midi.png", tf.expand_dims(midi, -1))

# 对midi补边，避免越界
midi = tf.pad(midi, tf.constant([[160, 160], [0, 0]]), mode='CONSTANT', constant_values=0)

# 频谱特征随机切分位置
pA = tf.random.uniform(shape=[], minval=0, maxval=320, dtype=tf.int32)
print(f"pA: {pA}")
# 标签特征随机切分位置
pB = tf.random.uniform(shape=[], minval=-160, maxval=160, dtype=tf.int32)
pB = 160 + pA + pB
# 制作label
label = tf.cast(pB + 160, tf.float32) / 320.

# 裁剪特征
feature = feature[pA: pA + 320, :, :]
print(feature.shape)
# midi = midi[pB: pB + 320, :]


# 拼接特征，检查结果
rst_mat = feature

# 保存图片
imageio.imwrite("demo.png", rst_mat / feature_max_value)

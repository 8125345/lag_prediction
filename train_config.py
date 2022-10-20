# 数据集整体数量控制，充分利用被下采样的数据集
mul_times = 1
# export PYTHONPATH=$PYTHONPATH:/data/projects/LabelModels
# # export PYTHONPATH=$PYTHONPATH:/data/projects/LabelModels/label_refiner
# export PYTHONPATH=$PYTHONPATH:/data/projects/LabelModels/label_refiner/models

# /Users/xyz/PycharmProjects/XYZ_projects/LabelModels/label_refiner
# 训练时使用的数据集
train_dataset = [

    ("maestro_new", 1. * mul_times),  # 70419
    ("xuanran_ori", 1. * mul_times), #12473


]
# 基础配置，如果train_config中存在同名变量，会用train_config变量覆盖basic_config配置
basic_config = {
    "model_root": "/deepiano_data/zhaoliang/project/lag_prediction/train_output_models",
    "train_log_root": "/deepiano_data/zhaoliang/project/lag_prediction/train_log",
    "train_comment_root": "/deepiano_data/zhaoliang/project/lag_prediction/train_comment",
}

# ==================================================================================
# ==================================================================================

# todo 目前数据增强配置参数在dataset_maker.py中，注意修改
# 本次训练配置

train_config = {
    "model_name": f"cornet_20221019_0",
    "gpuids": [0, 1],
    "train_batchsize": 48 + 16,
    "val_batchsize": 48 + 16,
    "add_maestro": True,
    "pos_weight": 1,
    "pretrain_model": "/deepiano_data/zhaoliang/project/lag_prediction/train_output_models/cornet_20221018_0.h5",  # cornet
    "model_structure": "cornet",
    "lr": 0.001 / (2 * 100),  # 初始学习率
    "dataset_path": "/data1/dataset/audio/rec_dataset/dataset_info_concat/dataset_20220916_0_train",
    # "rec_loss_fun": "weighted_bce",
    "rec_loss_fun": "mse",

    "comment": "频谱和midi延时预测模型，+-8，新模型v2",
}

# train_config = {
#     "model_name": f"cornet_20220926_3",
#     "gpuids": [0],
#     "train_batchsize": 48,
#     "val_batchsize": 48,
#     "add_maestro": True,
#     "pos_weight": 1,
#     "pretrain_model": "/data/projects/LabelModels/lag_prediction/train_output_models/cornet_20220926_2.h5",  # cornet
#     "model_structure": "cornet",
#     "lr": 0.001 / (3 * 10),  # 初始学习率
#     "dataset_path": "/data1/dataset/audio/rec_dataset/dataset_info_concat/dataset_20220916_0_train",  # todo，先使用次数据测试
#     # "rec_loss_fun": "weighted_bce",
#     "rec_loss_fun": "mse",
#
#     "comment": "频谱和midi延时预测模型，+-160数据增强，新模型v2",
# }
#
#
# train_config = {
#     "model_name": f"cornet_20220926_2",
#     "gpuids": [2, 3],
#     "train_batchsize": 48 * 2,
#     "val_batchsize": 48 * 2,
#     "add_maestro": True,
#     "pos_weight": 1,
#     "pretrain_model": "/data/projects/LabelModels/lag_prediction/train_output_models/cornet_20220926_2.h5",  # cornet
#     "model_structure": "cornet",
#     "lr": 0.001 / (2 * 100),  # 初始学习率
#     "dataset_path": "/data1/dataset/audio/rec_dataset/dataset_info_concat/dataset_20220916_0_train",  # todo，先使用次数据测试
#     # "rec_loss_fun": "weighted_bce",
#     "rec_loss_fun": "mse",
#
#     "comment": "频谱和midi延时预测模型，+-32，新模型v2",
# }

# train_config = {
#     "model_name": f"cornet_20220926_1",
#     "gpuids": [2],
#     "train_batchsize": 96 * 1,
#     "val_batchsize": 96 * 1,
#     "add_maestro": True,
#     "pos_weight": 1,
#     "pretrain_model": "/data/projects/LabelModels/lag_prediction/train_output_models/cornet_20220926_1.h5",  # cornet
#     "model_structure": "cornet",
#     "lr": 0.001 / (3 * 10),  # 初始学习率
#     "dataset_path": "/data1/dataset/audio/rec_dataset/dataset_info_concat/dataset_20220916_0_train",  # todo，先使用次数据测试
#     # "rec_loss_fun": "weighted_bce",
#     "rec_loss_fun": "mse",
#
#     "comment": "频谱和midi延时预测模型，降低数据增强",
# }

# train_config = {
#     "model_name": f"cornet_20220926_0",
#     "gpuids": [0, 2, 3],
#     "train_batchsize": 96 * 3,
#     "val_batchsize": 96 * 3,
#     "add_maestro": True,
#     "pos_weight": 1,
#     "pretrain_model": "/data/projects/LabelModels/lag_prediction/train_output_models/cornet_20220923_0.h5",  # cornet
#     "model_structure": "cornet",
#     "lr": 0.001 / (3 * 10),  # 初始学习率
#     "dataset_path": "/data1/dataset/audio/rec_dataset/dataset_info_concat/dataset_20220916_0_train",  # todo，先使用次数据测试
#     # "rec_loss_fun": "weighted_bce",
#     "rec_loss_fun": "mse",
#
#     "comment": "频谱和midi延时预测模型，矫正pB错误",
# }


# train_config = {
#     "model_name": f"cornet_20220923_0",
#     "gpuids": [0, 2, 3],
#     "train_batchsize": 96 * 3,
#     "val_batchsize": 96 * 3,
#     "add_maestro": True,
#     "pos_weight": 1,
#     "pretrain_model": "/data/projects/LabelModels/lag_prediction/train_output_models/cornet_20220923_0.h5",  # cornet
#     "model_structure": "cornet",
#     "lr": 0.001 / (3 * 10),  # 初始学习率
#     "dataset_path": "/data1/dataset/audio/rec_dataset/dataset_info_concat/dataset_20220916_0_train",  # todo，先使用次数据测试
#     # "rec_loss_fun": "weighted_bce",
#     "rec_loss_fun": "mse",
#
#     "comment": "频谱和midi延时预测模型",
# }

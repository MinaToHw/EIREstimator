import tensorflow as tf
import numpy as np
import os
import random


def get_structure_param_names():
    return ['RLR_min_delta', 'RLR_factor', 'RLR_patience', 'RLR_min_lr',
            'ES_min_delta', 'ES_patience', 'ES_restore_best_weights']


class TrainCallback(tf.keras.callbacks.Callback):
    """
    save_interval：每隔多少个 epoch 保存一次模型
    save_dir：保存完整模型的目录
    filename：保存文件时使用的前缀名
    """
    def __init__(self, save_interval, save_dir, filename):
        super(TrainCallback, self).__init__()
        self.save_interval = save_interval
        self.save_dir = save_dir
        self.filename = filename
        self.epoch_losses = []
        self.val_losses = []

        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        """每个epoch结束时的操作"""
        logs = logs or {}

        self.epoch_losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))


class CNN1D_BiLSTM_Train:
    def __init__(self,):
        self.callbacks = None
        self.RLR_min_delta = 0.0005
        self.RLR_factor = 0.8
        self.RLR_patience = 8
        self.RLR_min_lr = 1e-7
        self.ES_min_delta = 0.001
        self.ES_patience = 25
        self.ES_restore_best_weights = True

    def callback_setting(self, filename_savefile, save_dir):

        self.callbacks = [
            TrainCallback(100, save_dir, filename_savefile),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                min_delta=self.RLR_min_delta,
                factor=self.RLR_factor,
                patience=self.RLR_patience,
                min_lr=self.RLR_min_lr,
                verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                min_delta=self.ES_min_delta,
                patience=self.ES_patience,
                restore_best_weights=self.ES_restore_best_weights,
                verbose=1
            )
        ]

        return

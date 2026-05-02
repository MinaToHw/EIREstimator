#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# @Author: 
# @Date:   2026/3/19
# @Last Modified by:   
# @Last Modified time: 13:07
# ----------------------------------------------------------------------------
import tensorflow as tf
from keras import layers, Model


def get_network_param_names():
    return ['cnn1d_filters', 'cnn1d_kernel_sizes', 'biltsm_units', 'dropout_rates', 'dense_units', 'dense_l2']


class CNN1D_BiLSTM(Model):

    def __init__(self,):
        super(CNN1D_BiLSTM, self).__init__()
        self.output_dim = 4
        self.debug = False  # 添加调试开关

        self.cnn1d_filters = [16, 32, 64]
        self.cnn1d_kernel_sizes = [5, 3, 3]
        self.biltsm_units = [16]
        self.dropout_rates = [0.2, 0.2, 0.2, 0.2]
        self.dense_units = [256, 128]
        self.dense_l2 = [0.0005, 0.0005]

        self.cnn1d_bilstm = tf.keras.Sequential([
            layers.Conv1D(filters=self.cnn1d_filters[0], kernel_size=self.cnn1d_kernel_sizes[0], activation='relu', padding='same', input_shape=(256, 1)),
            layers.BatchNormalization(),

            layers.Conv1D(filters=self.cnn1d_filters[1], kernel_size=self.cnn1d_kernel_sizes[1], activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2, strides=2),

            layers.Conv1D(filters=self.cnn1d_filters[2], kernel_size=self.cnn1d_kernel_sizes[2], activation='relu', padding='same'),
            layers.BatchNormalization(),

            layers.Dropout(self.dropout_rates[0]),
            layers.MaxPooling1D(pool_size=2, strides=2),

            layers.Bidirectional(layers.LSTM(self.biltsm_units[0], return_sequences=True, activation='tanh')),
            layers.BatchNormalization(),
            layers.Dropout(self.dropout_rates[1]),

            layers.Flatten(),

            layers.Dense(self.dense_units[0], activation='relu', kernel_regularizer=tf.keras.regularizers.l2(self.dense_l2[0])),
            layers.Dropout(self.dropout_rates[2]),
            layers.Dense(self.dense_units[1], activation='relu', kernel_regularizer=tf.keras.regularizers.l2(self.dense_l2[1])),
            layers.Dropout(self.dropout_rates[3]),
            layers.Dense(self.output_dim, activation='linear'),
        ])

    def call(self, x, training=False):
        """前向传播函数，支持逐层输出调试"""
        if len(x.shape) == 2:
            x = tf.expand_dims(x, axis=-1)

        if self.debug:
            print(f"\n{'=' * 50}")
            print(f"输入形状: {x.shape}")
            print(f"{'=' * 50}")

            # 逐层传递并打印形状
            for i, layer in enumerate(self.cnn1d_bilstm.layers):
                x = layer(x, training=training)
                print(f"层 {i}: {layer.name}")
                print(f"  输出形状: {x.shape}")
                if hasattr(layer, 'filters'):
                    print(f"  过滤器: {layer.filters}")
                print("-" * 30)
            return x
        else:
            return self.cnn1d_bilstm(x, training=training)

    def get_config(self):
        """用于模型保存的配置"""
        config = super(CNN1D_BiLSTM, self).get_config()
        config.update({
            'output_dim': self.output_dim,
        })
        return config

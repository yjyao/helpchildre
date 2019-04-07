Using seed 874450
CNN(
  (encoder): Sequential(
    (0): ConvLayer(
      (0): Conv2d(3, 96, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
      (1): ReLU()
      (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (3): Dropout(p=0.2)
    )
    (1): ConvLayer(
      (0): Conv2d(96, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): ReLU()
      (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (3): Dropout(p=0.5)
    )
  )
  (decoder): Sequential(
    (0): BatchNorm1d(12288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (1): FcLayer(
      (0): Linear(in_features=12288, out_features=256, bias=True)
      (1): ReLU()
    )
    (2): FcLayer(
      (0): Linear(in_features=256, out_features=128, bias=True)
      (1): ReLU()
    )
    (3): FcLayer(
      (0): Linear(in_features=128, out_features=10, bias=True)
    )
  )
)
Using 3,378,122 parameters
[('encoder.0.0.weight', 7200), ('encoder.0.0.bias', 96), ('encoder.1.0.weight', 165888), ('encoder.1.0.bias', 192), ('decoder.0.weight', 12288), ('decoder.0.bias', 12288), ('decoder.1.0.weight', 3145728), ('decoder.1.0.bias', 256), ('decoder.2.0.weight', 32768), ('decoder.2.0.bias', 128), ('decoder.3.0.weight', 1280), ('decoder.3.0.bias', 10)]
TRAINING
==============================
Epoches: 50
Batch size: 128
Learning rate: 0.001

[02:26:46] Epoch 1 [         ] loss: 1.753
[02:27:11] Epoch 1 [>        ] loss: 1.430
[02:27:35] Epoch 1 [=>       ] loss: 1.341
[02:28:00] Epoch 1 [==>      ] loss: 1.266
[02:28:23] Epoch 1 [===>     ] loss: 1.215
[02:28:46] Epoch 1 [====>    ] loss: 1.183
[02:29:09] Epoch 1 [=====>   ] loss: 1.174
[02:29:31] Epoch 1 [======>  ] loss: 1.130
[02:29:55] Epoch 1 [=======> ] loss: 1.077
[02:30:18] Epoch 1 [========>] loss: 1.072
Test accuracy of the cnn on the 50000 train images: 67.84%
Test accuracy of the cnn on the 10000 test images: 63.85%
[02:31:34] Epoch 2 [         ] loss: 0.992
[02:31:59] Epoch 2 [>        ] loss: 0.965
[02:32:29] Epoch 2 [=>       ] loss: 0.940
[02:32:58] Epoch 2 [==>      ] loss: 0.977
[02:33:27] Epoch 2 [===>     ] loss: 0.935
[02:33:56] Epoch 2 [====>    ] loss: 0.950
[02:34:24] Epoch 2 [=====>   ] loss: 0.908
[02:34:53] Epoch 2 [======>  ] loss: 0.916
[02:35:21] Epoch 2 [=======> ] loss: 0.911
[02:35:50] Epoch 2 [========>] loss: 0.917
Test accuracy of the cnn on the 50000 train images: 75.28%
Test accuracy of the cnn on the 10000 test images: 70.18%
[02:38:13] Epoch 3 [         ] loss: 0.813
[02:38:42] Epoch 3 [>        ] loss: 0.801
[02:39:12] Epoch 3 [=>       ] loss: 0.761
[02:39:41] Epoch 3 [==>      ] loss: 0.793
[02:40:09] Epoch 3 [===>     ] loss: 0.793
[02:40:39] Epoch 3 [====>    ] loss: 0.780
[02:41:08] Epoch 3 [=====>   ] loss: 0.797
[02:41:37] Epoch 3 [======>  ] loss: 0.790
[02:42:06] Epoch 3 [=======> ] loss: 0.780
[02:42:36] Epoch 3 [========>] loss: 0.791
Test accuracy of the cnn on the 50000 train images: 80.11%
Test accuracy of the cnn on the 10000 test images: 73.07%
[02:44:59] Epoch 4 [         ] loss: 0.659
[02:45:30] Epoch 4 [>        ] loss: 0.667
[02:46:01] Epoch 4 [=>       ] loss: 0.683
[02:46:32] Epoch 4 [==>      ] loss: 0.670
[02:47:03] Epoch 4 [===>     ] loss: 0.682
[02:47:34] Epoch 4 [====>    ] loss: 0.679
[02:48:05] Epoch 4 [=====>   ] loss: 0.681
[02:48:36] Epoch 4 [======>  ] loss: 0.670
[02:49:07] Epoch 4 [=======> ] loss: 0.699
[02:49:38] Epoch 4 [========>] loss: 0.678
Test accuracy of the cnn on the 50000 train images: 84.16%
Test accuracy of the cnn on the 10000 test images: 74.73%
[02:52:01] Epoch 5 [         ] loss: 0.572
[02:52:31] Epoch 5 [>        ] loss: 0.563
[02:53:02] Epoch 5 [=>       ] loss: 0.587
[02:53:33] Epoch 5 [==>      ] loss: 0.584
[02:54:03] Epoch 5 [===>     ] loss: 0.604
[02:54:34] Epoch 5 [====>    ] loss: 0.591
[02:55:04] Epoch 5 [=====>   ] loss: 0.594
[02:55:34] Epoch 5 [======>  ] loss: 0.625
[02:56:05] Epoch 5 [=======> ] loss: 0.606
[02:56:36] Epoch 5 [========>] loss: 0.606
Test accuracy of the cnn on the 50000 train images: 87.94%
Test accuracy of the cnn on the 10000 test images: 75.72%
[02:58:59] Epoch 6 [         ] loss: 0.500
[02:59:29] Epoch 6 [>        ] loss: 0.510
[03:00:00] Epoch 6 [=>       ] loss: 0.498
[03:00:31] Epoch 6 [==>      ] loss: 0.509
[03:01:02] Epoch 6 [===>     ] loss: 0.530
[03:01:32] Epoch 6 [====>    ] loss: 0.514
[03:02:03] Epoch 6 [=====>   ] loss: 0.552
[03:02:34] Epoch 6 [======>  ] loss: 0.537
[03:03:05] Epoch 6 [=======> ] loss: 0.537
[03:03:36] Epoch 6 [========>] loss: 0.544
Test accuracy of the cnn on the 50000 train images: 89.80%
Test accuracy of the cnn on the 10000 test images: 76.42%
[03:05:57] Epoch 7 [         ] loss: 0.399
[03:06:28] Epoch 7 [>        ] loss: 0.416
[03:06:59] Epoch 7 [=>       ] loss: 0.437
[03:07:30] Epoch 7 [==>      ] loss: 0.475
[03:08:01] Epoch 7 [===>     ] loss: 0.469
[03:08:31] Epoch 7 [====>    ] loss: 0.479
[03:09:01] Epoch 7 [=====>   ] loss: 0.466
[03:09:32] Epoch 7 [======>  ] loss: 0.463
[03:10:02] Epoch 7 [=======> ] loss: 0.474
[03:10:34] Epoch 7 [========>] loss: 0.504
Test accuracy of the cnn on the 50000 train images: 91.28%
Test accuracy of the cnn on the 10000 test images: 75.37%
[03:12:59] Epoch 8 [         ] loss: 0.376
[03:13:30] Epoch 8 [>        ] loss: 0.398
[03:14:01] Epoch 8 [=>       ] loss: 0.393
[03:14:31] Epoch 8 [==>      ] loss: 0.409
[03:15:02] Epoch 8 [===>     ] loss: 0.383
[03:15:33] Epoch 8 [====>    ] loss: 0.395
[03:16:04] Epoch 8 [=====>   ] loss: 0.415
[03:16:35] Epoch 8 [======>  ] loss: 0.409
[03:17:06] Epoch 8 [=======> ] loss: 0.424
[03:17:37] Epoch 8 [========>] loss: 0.428
Test accuracy of the cnn on the 50000 train images: 94.43%
Test accuracy of the cnn on the 10000 test images: 76.70%
[03:19:58] Epoch 9 [         ] loss: 0.325
[03:20:29] Epoch 9 [>        ] loss: 0.333
[03:20:59] Epoch 9 [=>       ] loss: 0.339
[03:21:30] Epoch 9 [==>      ] loss: 0.344
[03:22:00] Epoch 9 [===>     ] loss: 0.352
[03:22:30] Epoch 9 [====>    ] loss: 0.369
[03:23:01] Epoch 9 [=====>   ] loss: 0.361
[03:23:31] Epoch 9 [======>  ] loss: 0.390
[03:24:01] Epoch 9 [=======> ] loss: 0.374
[03:24:31] Epoch 9 [========>] loss: 0.370
Test accuracy of the cnn on the 50000 train images: 94.87%
Test accuracy of the cnn on the 10000 test images: 76.60%
[03:26:53] Epoch 10 [         ] loss: 0.298
[03:27:24] Epoch 10 [>        ] loss: 0.278
[03:27:54] Epoch 10 [=>       ] loss: 0.305
[03:28:24] Epoch 10 [==>      ] loss: 0.325
[03:28:55] Epoch 10 [===>     ] loss: 0.283
[03:29:25] Epoch 10 [====>    ] loss: 0.309
[03:29:56] Epoch 10 [=====>   ] loss: 0.349
[03:30:26] Epoch 10 [======>  ] loss: 0.340
[03:30:57] Epoch 10 [=======> ] loss: 0.336
[03:31:27] Epoch 10 [========>] loss: 0.344
Test accuracy of the cnn on the 50000 train images: 96.23%
Test accuracy of the cnn on the 10000 test images: 76.28%
[03:33:49] Epoch 11 [         ] loss: 0.247
[03:34:20] Epoch 11 [>        ] loss: 0.265
[03:34:51] Epoch 11 [=>       ] loss: 0.288
[03:35:22] Epoch 11 [==>      ] loss: 0.284
[03:35:52] Epoch 11 [===>     ] loss: 0.284
[03:36:23] Epoch 11 [====>    ] loss: 0.293
[03:36:54] Epoch 11 [=====>   ] loss: 0.286
[03:37:24] Epoch 11 [======>  ] loss: 0.283
[03:37:55] Epoch 11 [=======> ] loss: 0.310
[03:38:26] Epoch 11 [========>] loss: 0.334
Test accuracy of the cnn on the 50000 train images: 96.25%
Test accuracy of the cnn on the 10000 test images: 76.43%
[03:40:47] Epoch 12 [         ] loss: 0.236
[03:41:18] Epoch 12 [>        ] loss: 0.251
[03:41:48] Epoch 12 [=>       ] loss: 0.239
[03:42:19] Epoch 12 [==>      ] loss: 0.250
[03:42:49] Epoch 12 [===>     ] loss: 0.264
[03:43:20] Epoch 12 [====>    ] loss: 0.292
[03:43:50] Epoch 12 [=====>   ] loss: 0.264
[03:44:20] Epoch 12 [======>  ] loss: 0.261
[03:44:51] Epoch 12 [=======> ] loss: 0.274
[03:45:21] Epoch 12 [========>] loss: 0.296
Test accuracy of the cnn on the 50000 train images: 97.73%
Test accuracy of the cnn on the 10000 test images: 77.48%
[03:47:43] Epoch 13 [         ] loss: 0.221
[03:48:14] Epoch 13 [>        ] loss: 0.238
[03:48:45] Epoch 13 [=>       ] loss: 0.237
[03:49:16] Epoch 13 [==>      ] loss: 0.241
[03:49:47] Epoch 13 [===>     ] loss: 0.242
[03:50:18] Epoch 13 [====>    ] loss: 0.241
[03:50:49] Epoch 13 [=====>   ] loss: 0.263
[03:51:19] Epoch 13 [======>  ] loss: 0.263
[03:51:50] Epoch 13 [=======> ] loss: 0.270
[03:52:20] Epoch 13 [========>] loss: 0.282
Test accuracy of the cnn on the 50000 train images: 97.63%
Test accuracy of the cnn on the 10000 test images: 76.91%
[03:54:42] Epoch 14 [         ] loss: 0.191
[03:55:13] Epoch 14 [>        ] loss: 0.199
[03:55:43] Epoch 14 [=>       ] loss: 0.222
[03:56:13] Epoch 14 [==>      ] loss: 0.240
[03:56:43] Epoch 14 [===>     ] loss: 0.230
[03:57:14] Epoch 14 [====>    ] loss: 0.225
[03:57:44] Epoch 14 [=====>   ] loss: 0.262
[03:58:14] Epoch 14 [======>  ] loss: 0.245
[03:58:45] Epoch 14 [=======> ] loss: 0.251
[03:59:15] Epoch 14 [========>] loss: 0.259
Test accuracy of the cnn on the 50000 train images: 98.26%
Test accuracy of the cnn on the 10000 test images: 76.88%
[04:01:37] Epoch 15 [         ] loss: 0.184
[04:02:07] Epoch 15 [>        ] loss: 0.185
[04:02:37] Epoch 15 [=>       ] loss: 0.185
[04:03:07] Epoch 15 [==>      ] loss: 0.199
[04:03:38] Epoch 15 [===>     ] loss: 0.218
[04:04:08] Epoch 15 [====>    ] loss: 0.205
[04:04:38] Epoch 15 [=====>   ] loss: 0.223
[04:05:08] Epoch 15 [======>  ] loss: 0.233
[04:05:39] Epoch 15 [=======> ] loss: 0.250
[04:06:09] Epoch 15 [========>] loss: 0.247
Test accuracy of the cnn on the 50000 train images: 98.22%
Test accuracy of the cnn on the 10000 test images: 76.70%
[04:08:31] Epoch 16 [         ] loss: 0.183
[04:09:01] Epoch 16 [>        ] loss: 0.186
[04:09:31] Epoch 16 [=>       ] loss: 0.181
[04:10:02] Epoch 16 [==>      ] loss: 0.191
[04:10:32] Epoch 16 [===>     ] loss: 0.188
[04:11:02] Epoch 16 [====>    ] loss: 0.217
[04:11:31] Epoch 16 [=====>   ] loss: 0.224
[04:11:58] Epoch 16 [======>  ] loss: 0.197
[04:12:24] Epoch 16 [=======> ] loss: 0.210
[04:12:50] Epoch 16 [========>] loss: 0.229
Test accuracy of the cnn on the 50000 train images: 98.91%
Test accuracy of the cnn on the 10000 test images: 76.96%
[04:15:07] Epoch 17 [         ] loss: 0.189
[04:15:34] Epoch 17 [>        ] loss: 0.163
[04:16:00] Epoch 17 [=>       ] loss: 0.157
[04:16:26] Epoch 17 [==>      ] loss: 0.191
[04:16:52] Epoch 17 [===>     ] loss: 0.187
[04:17:18] Epoch 17 [====>    ] loss: 0.193
[04:17:45] Epoch 17 [=====>   ] loss: 0.193
[04:18:11] Epoch 17 [======>  ] loss: 0.209
[04:18:37] Epoch 17 [=======> ] loss: 0.211
[04:19:03] Epoch 17 [========>] loss: 0.198
Test accuracy of the cnn on the 50000 train images: 98.93%
Test accuracy of the cnn on the 10000 test images: 77.06%
[04:21:23] Epoch 18 [         ] loss: 0.169
[04:21:49] Epoch 18 [>        ] loss: 0.154
[04:22:16] Epoch 18 [=>       ] loss: 0.156
[04:22:42] Epoch 18 [==>      ] loss: 0.175
[04:23:08] Epoch 18 [===>     ] loss: 0.159
[04:23:34] Epoch 18 [====>    ] loss: 0.180
[04:24:00] Epoch 18 [=====>   ] loss: 0.187
[04:24:27] Epoch 18 [======>  ] loss: 0.186
[04:24:52] Epoch 18 [=======> ] loss: 0.207
[04:25:19] Epoch 18 [========>] loss: 0.193
Test accuracy of the cnn on the 50000 train images: 99.03%
Test accuracy of the cnn on the 10000 test images: 76.90%
[04:27:36] Epoch 19 [         ] loss: 0.155
[04:28:02] Epoch 19 [>        ] loss: 0.155
[04:28:28] Epoch 19 [=>       ] loss: 0.161
[04:28:54] Epoch 19 [==>      ] loss: 0.172
[04:29:20] Epoch 19 [===>     ] loss: 0.167
[04:29:46] Epoch 19 [====>    ] loss: 0.161
[04:30:12] Epoch 19 [=====>   ] loss: 0.181
[04:30:38] Epoch 19 [======>  ] loss: 0.180
[04:31:04] Epoch 19 [=======> ] loss: 0.185
[04:31:31] Epoch 19 [========>] loss: 0.208
Test accuracy of the cnn on the 50000 train images: 99.23%
Test accuracy of the cnn on the 10000 test images: 77.59%
[04:33:48] Epoch 20 [         ] loss: 0.153
[04:34:14] Epoch 20 [>        ] loss: 0.155
[04:34:40] Epoch 20 [=>       ] loss: 0.158
[04:35:06] Epoch 20 [==>      ] loss: 0.154
[04:35:31] Epoch 20 [===>     ] loss: 0.160
[04:35:57] Epoch 20 [====>    ] loss: 0.174
[04:36:23] Epoch 20 [=====>   ] loss: 0.184
[04:36:49] Epoch 20 [======>  ] loss: 0.169
[04:37:15] Epoch 20 [=======> ] loss: 0.156
[04:37:41] Epoch 20 [========>] loss: 0.193
Test accuracy of the cnn on the 50000 train images: 99.16%
Test accuracy of the cnn on the 10000 test images: 76.86%
[04:39:59] Epoch 21 [         ] loss: 0.130
[04:40:25] Epoch 21 [>        ] loss: 0.137
[04:40:51] Epoch 21 [=>       ] loss: 0.154
[04:41:17] Epoch 21 [==>      ] loss: 0.153
[04:41:43] Epoch 21 [===>     ] loss: 0.160
[04:42:10] Epoch 21 [====>    ] loss: 0.162
[04:42:35] Epoch 21 [=====>   ] loss: 0.176
[04:43:02] Epoch 21 [======>  ] loss: 0.144
[04:43:28] Epoch 21 [=======> ] loss: 0.167
[04:43:54] Epoch 21 [========>] loss: 0.180
Test accuracy of the cnn on the 50000 train images: 99.33%
Test accuracy of the cnn on the 10000 test images: 77.23%
[04:46:11] Epoch 22 [         ] loss: 0.152
[04:46:37] Epoch 22 [>        ] loss: 0.147
[04:47:04] Epoch 22 [=>       ] loss: 0.154
[04:47:30] Epoch 22 [==>      ] loss: 0.147
[04:47:56] Epoch 22 [===>     ] loss: 0.162
[04:48:22] Epoch 22 [====>    ] loss: 0.163
[04:48:48] Epoch 22 [=====>   ] loss: 0.148
[04:49:14] Epoch 22 [======>  ] loss: 0.173
[04:49:40] Epoch 22 [=======> ] loss: 0.155
[04:50:06] Epoch 22 [========>] loss: 0.172
Test accuracy of the cnn on the 50000 train images: 99.42%
Test accuracy of the cnn on the 10000 test images: 77.37%
[04:52:23] Epoch 23 [         ] loss: 0.131
[04:52:50] Epoch 23 [>        ] loss: 0.121
[04:53:16] Epoch 23 [=>       ] loss: 0.132
[04:53:42] Epoch 23 [==>      ] loss: 0.133
[04:54:08] Epoch 23 [===>     ] loss: 0.138
[04:54:34] Epoch 23 [====>    ] loss: 0.138
[04:55:00] Epoch 23 [=====>   ] loss: 0.143
[04:55:26] Epoch 23 [======>  ] loss: 0.149
[04:55:52] Epoch 23 [=======> ] loss: 0.146
[04:56:18] Epoch 23 [========>] loss: 0.158
Test accuracy of the cnn on the 50000 train images: 99.50%
Test accuracy of the cnn on the 10000 test images: 77.48%
[04:58:35] Epoch 24 [         ] loss: 0.125
[04:59:01] Epoch 24 [>        ] loss: 0.129
[04:59:27] Epoch 24 [=>       ] loss: 0.135
[04:59:53] Epoch 24 [==>      ] loss: 0.143
[05:00:19] Epoch 24 [===>     ] loss: 0.130
[05:00:45] Epoch 24 [====>    ] loss: 0.141
[05:01:11] Epoch 24 [=====>   ] loss: 0.139
[05:01:37] Epoch 24 [======>  ] loss: 0.147
[05:02:03] Epoch 24 [=======> ] loss: 0.143
[05:02:29] Epoch 24 [========>] loss: 0.151
Test accuracy of the cnn on the 50000 train images: 99.60%
Test accuracy of the cnn on the 10000 test images: 77.29%
[05:04:46] Epoch 25 [         ] loss: 0.143
[05:05:12] Epoch 25 [>        ] loss: 0.143
[05:05:39] Epoch 25 [=>       ] loss: 0.128
[05:06:05] Epoch 25 [==>      ] loss: 0.119
[05:06:31] Epoch 25 [===>     ] loss: 0.152
[05:06:58] Epoch 25 [====>    ] loss: 0.133
[05:07:24] Epoch 25 [=====>   ] loss: 0.140
[05:07:50] Epoch 25 [======>  ] loss: 0.172
[05:08:16] Epoch 25 [=======> ] loss: 0.160
[05:08:42] Epoch 25 [========>] loss: 0.146
Test accuracy of the cnn on the 50000 train images: 99.31%
Test accuracy of the cnn on the 10000 test images: 77.44%
[05:11:00] Epoch 26 [         ] loss: 0.140
[05:11:26] Epoch 26 [>        ] loss: 0.141
[05:11:52] Epoch 26 [=>       ] loss: 0.134
[05:12:18] Epoch 26 [==>      ] loss: 0.128
[05:12:44] Epoch 26 [===>     ] loss: 0.145
[05:13:10] Epoch 26 [====>    ] loss: 0.137
[05:13:36] Epoch 26 [=====>   ] loss: 0.136
[05:14:02] Epoch 26 [======>  ] loss: 0.132
[05:14:28] Epoch 26 [=======> ] loss: 0.124
[05:14:54] Epoch 26 [========>] loss: 0.126
Test accuracy of the cnn on the 50000 train images: 99.59%
Test accuracy of the cnn on the 10000 test images: 77.08%
[05:17:11] Epoch 27 [         ] loss: 0.131
[05:17:37] Epoch 27 [>        ] loss: 0.119
[05:18:03] Epoch 27 [=>       ] loss: 0.113
[05:18:29] Epoch 27 [==>      ] loss: 0.124
[05:18:55] Epoch 27 [===>     ] loss: 0.143
[05:19:22] Epoch 27 [====>    ] loss: 0.132
[05:19:48] Epoch 27 [=====>   ] loss: 0.118
[05:20:14] Epoch 27 [======>  ] loss: 0.130
[05:20:40] Epoch 27 [=======> ] loss: 0.144
[05:21:06] Epoch 27 [========>] loss: 0.128
Test accuracy of the cnn on the 50000 train images: 99.67%
Test accuracy of the cnn on the 10000 test images: 77.30%
[05:23:24] Epoch 28 [         ] loss: 0.120
[05:23:49] Epoch 28 [>        ] loss: 0.097
[05:24:15] Epoch 28 [=>       ] loss: 0.118
[05:24:42] Epoch 28 [==>      ] loss: 0.119
[05:25:08] Epoch 28 [===>     ] loss: 0.138
[05:25:34] Epoch 28 [====>    ] loss: 0.122
[05:26:00] Epoch 28 [=====>   ] loss: 0.124
[05:26:26] Epoch 28 [======>  ] loss: 0.118
[05:26:51] Epoch 28 [=======> ] loss: 0.142
[05:27:18] Epoch 28 [========>] loss: 0.152
Test accuracy of the cnn on the 50000 train images: 99.49%
Test accuracy of the cnn on the 10000 test images: 77.28%
[05:29:35] Epoch 29 [         ] loss: 0.120
[05:30:01] Epoch 29 [>        ] loss: 0.110
[05:30:26] Epoch 29 [=>       ] loss: 0.125
[05:30:52] Epoch 29 [==>      ] loss: 0.100
[05:31:18] Epoch 29 [===>     ] loss: 0.117
[05:31:44] Epoch 29 [====>    ] loss: 0.139
[05:32:10] Epoch 29 [=====>   ] loss: 0.119
[05:32:36] Epoch 29 [======>  ] loss: 0.142
[05:33:02] Epoch 29 [=======> ] loss: 0.137
[05:33:28] Epoch 29 [========>] loss: 0.134
Test accuracy of the cnn on the 50000 train images: 99.63%
Test accuracy of the cnn on the 10000 test images: 77.25%
[05:35:45] Epoch 30 [         ] loss: 0.097
[05:36:11] Epoch 30 [>        ] loss: 0.111
[05:36:37] Epoch 30 [=>       ] loss: 0.115
[05:37:03] Epoch 30 [==>      ] loss: 0.105
[05:37:29] Epoch 30 [===>     ] loss: 0.118
[05:37:55] Epoch 30 [====>    ] loss: 0.108
[05:38:21] Epoch 30 [=====>   ] loss: 0.121
[05:38:47] Epoch 30 [======>  ] loss: 0.119
[05:39:13] Epoch 30 [=======> ] loss: 0.116
[05:39:39] Epoch 30 [========>] loss: 0.126
Test accuracy of the cnn on the 50000 train images: 99.45%
Test accuracy of the cnn on the 10000 test images: 76.60%
[05:41:56] Epoch 31 [         ] loss: 0.134
[05:42:22] Epoch 31 [>        ] loss: 0.110
[05:42:48] Epoch 31 [=>       ] loss: 0.112
[05:43:14] Epoch 31 [==>      ] loss: 0.112
[05:43:39] Epoch 31 [===>     ] loss: 0.113
[05:44:05] Epoch 31 [====>    ] loss: 0.114
[05:44:31] Epoch 31 [=====>   ] loss: 0.115
[05:44:57] Epoch 31 [======>  ] loss: 0.126
[05:45:23] Epoch 31 [=======> ] loss: 0.116
[05:45:49] Epoch 31 [========>] loss: 0.133
Test accuracy of the cnn on the 50000 train images: 99.71%
Test accuracy of the cnn on the 10000 test images: 77.12%
[05:48:07] Epoch 32 [         ] loss: 0.096
[05:48:33] Epoch 32 [>        ] loss: 0.121
[05:48:58] Epoch 32 [=>       ] loss: 0.108
[05:49:24] Epoch 32 [==>      ] loss: 0.100
[05:49:50] Epoch 32 [===>     ] loss: 0.096
[05:50:16] Epoch 32 [====>    ] loss: 0.126
[05:50:42] Epoch 32 [=====>   ] loss: 0.140
[05:51:08] Epoch 32 [======>  ] loss: 0.111
[05:51:34] Epoch 32 [=======> ] loss: 0.113
[05:52:00] Epoch 32 [========>] loss: 0.123
Test accuracy of the cnn on the 50000 train images: 99.62%
Test accuracy of the cnn on the 10000 test images: 77.10%
[05:54:17] Epoch 33 [         ] loss: 0.095
[05:54:43] Epoch 33 [>        ] loss: 0.097
[05:55:09] Epoch 33 [=>       ] loss: 0.096
[05:55:35] Epoch 33 [==>      ] loss: 0.107
[05:56:01] Epoch 33 [===>     ] loss: 0.126
[05:56:27] Epoch 33 [====>    ] loss: 0.121
[05:56:53] Epoch 33 [=====>   ] loss: 0.116
[05:57:18] Epoch 33 [======>  ] loss: 0.127
[05:57:44] Epoch 33 [=======> ] loss: 0.121
[05:58:10] Epoch 33 [========>] loss: 0.132
Test accuracy of the cnn on the 50000 train images: 99.57%
Test accuracy of the cnn on the 10000 test images: 77.20%
[06:00:28] Epoch 34 [         ] loss: 0.099
[06:00:53] Epoch 34 [>        ] loss: 0.110
[06:01:20] Epoch 34 [=>       ] loss: 0.117
[06:01:46] Epoch 34 [==>      ] loss: 0.099
[06:02:12] Epoch 34 [===>     ] loss: 0.108
[06:02:38] Epoch 34 [====>    ] loss: 0.115
[06:03:03] Epoch 34 [=====>   ] loss: 0.108
[06:03:30] Epoch 34 [======>  ] loss: 0.098
[06:03:55] Epoch 34 [=======> ] loss: 0.111
[06:04:21] Epoch 34 [========>] loss: 0.115
Test accuracy of the cnn on the 50000 train images: 99.80%
Test accuracy of the cnn on the 10000 test images: 77.45%
[06:06:39] Epoch 35 [         ] loss: 0.100
[06:07:05] Epoch 35 [>        ] loss: 0.095
[06:07:31] Epoch 35 [=>       ] loss: 0.101
[06:07:56] Epoch 35 [==>      ] loss: 0.100
[06:08:22] Epoch 35 [===>     ] loss: 0.112
[06:08:48] Epoch 35 [====>    ] loss: 0.102
[06:09:14] Epoch 35 [=====>   ] loss: 0.102
[06:09:40] Epoch 35 [======>  ] loss: 0.118
[06:10:06] Epoch 35 [=======> ] loss: 0.109
[06:10:33] Epoch 35 [========>] loss: 0.099
Test accuracy of the cnn on the 50000 train images: 99.80%
Test accuracy of the cnn on the 10000 test images: 78.01%
[06:12:49] Epoch 36 [         ] loss: 0.097
[06:13:16] Epoch 36 [>        ] loss: 0.090
[06:13:41] Epoch 36 [=>       ] loss: 0.079
[06:14:07] Epoch 36 [==>      ] loss: 0.089
[06:14:33] Epoch 36 [===>     ] loss: 0.106
[06:14:59] Epoch 36 [====>    ] loss: 0.115
[06:15:25] Epoch 36 [=====>   ] loss: 0.119
[06:15:52] Epoch 36 [======>  ] loss: 0.097
[06:16:18] Epoch 36 [=======> ] loss: 0.104
[06:16:44] Epoch 36 [========>] loss: 0.107
Test accuracy of the cnn on the 50000 train images: 99.54%
Test accuracy of the cnn on the 10000 test images: 77.28%
[06:19:01] Epoch 37 [         ] loss: 0.092
[06:19:28] Epoch 37 [>        ] loss: 0.099
[06:19:54] Epoch 37 [=>       ] loss: 0.106
[06:20:20] Epoch 37 [==>      ] loss: 0.106
[06:20:46] Epoch 37 [===>     ] loss: 0.104
[06:21:12] Epoch 37 [====>    ] loss: 0.095
[06:21:38] Epoch 37 [=====>   ] loss: 0.110
[06:22:04] Epoch 37 [======>  ] loss: 0.116
[06:22:30] Epoch 37 [=======> ] loss: 0.092
[06:22:56] Epoch 37 [========>] loss: 0.103
Test accuracy of the cnn on the 50000 train images: 99.78%
Test accuracy of the cnn on the 10000 test images: 77.45%
[06:25:14] Epoch 38 [         ] loss: 0.081
[06:25:40] Epoch 38 [>        ] loss: 0.104
[06:26:06] Epoch 38 [=>       ] loss: 0.093
[06:26:32] Epoch 38 [==>      ] loss: 0.095
[06:26:58] Epoch 38 [===>     ] loss: 0.102
[06:27:24] Epoch 38 [====>    ] loss: 0.102
[06:27:50] Epoch 38 [=====>   ] loss: 0.104
[06:28:16] Epoch 38 [======>  ] loss: 0.117
[06:28:41] Epoch 38 [=======> ] loss: 0.104
[06:29:08] Epoch 38 [========>] loss: 0.110
Test accuracy of the cnn on the 50000 train images: 99.76%
Test accuracy of the cnn on the 10000 test images: 77.33%
[06:31:26] Epoch 39 [         ] loss: 0.090
[06:31:52] Epoch 39 [>        ] loss: 0.107
[06:32:18] Epoch 39 [=>       ] loss: 0.102
[06:32:44] Epoch 39 [==>      ] loss: 0.088
[06:33:10] Epoch 39 [===>     ] loss: 0.093
[06:33:36] Epoch 39 [====>    ] loss: 0.110
[06:34:02] Epoch 39 [=====>   ] loss: 0.121
[06:34:28] Epoch 39 [======>  ] loss: 0.116
[06:34:54] Epoch 39 [=======> ] loss: 0.114
[06:35:19] Epoch 39 [========>] loss: 0.106
Test accuracy of the cnn on the 50000 train images: 99.77%
Test accuracy of the cnn on the 10000 test images: 77.44%
[06:37:37] Epoch 40 [         ] loss: 0.088
[06:38:03] Epoch 40 [>        ] loss: 0.089
[06:38:29] Epoch 40 [=>       ] loss: 0.097
[06:38:55] Epoch 40 [==>      ] loss: 0.089
[06:39:20] Epoch 40 [===>     ] loss: 0.111
[06:39:46] Epoch 40 [====>    ] loss: 0.086
[06:40:12] Epoch 40 [=====>   ] loss: 0.084
[06:40:39] Epoch 40 [======>  ] loss: 0.110
[06:41:05] Epoch 40 [=======> ] loss: 0.117
[06:41:31] Epoch 40 [========>] loss: 0.096
Test accuracy of the cnn on the 50000 train images: 99.72%
Test accuracy of the cnn on the 10000 test images: 77.30%
[06:43:48] Epoch 41 [         ] loss: 0.076
[06:44:13] Epoch 41 [>        ] loss: 0.076
[06:44:39] Epoch 41 [=>       ] loss: 0.082
[06:45:05] Epoch 41 [==>      ] loss: 0.093
[06:45:31] Epoch 41 [===>     ] loss: 0.088
[06:45:57] Epoch 41 [====>    ] loss: 0.098
[06:46:23] Epoch 41 [=====>   ] loss: 0.101
[06:46:50] Epoch 41 [======>  ] loss: 0.086
[06:47:16] Epoch 41 [=======> ] loss: 0.091
[06:47:42] Epoch 41 [========>] loss: 0.081
Test accuracy of the cnn on the 50000 train images: 99.73%
Test accuracy of the cnn on the 10000 test images: 77.85%
[06:49:59] Epoch 42 [         ] loss: 0.078
[06:50:25] Epoch 42 [>        ] loss: 0.093
[06:50:51] Epoch 42 [=>       ] loss: 0.085
[06:51:17] Epoch 42 [==>      ] loss: 0.089
[06:51:43] Epoch 42 [===>     ] loss: 0.090
[06:52:09] Epoch 42 [====>    ] loss: 0.102
[06:52:35] Epoch 42 [=====>   ] loss: 0.097
[06:53:01] Epoch 42 [======>  ] loss: 0.099
[06:53:27] Epoch 42 [=======> ] loss: 0.099
[06:53:53] Epoch 42 [========>] loss: 0.086
Test accuracy of the cnn on the 50000 train images: 99.76%
Test accuracy of the cnn on the 10000 test images: 77.87%
[06:56:11] Epoch 43 [         ] loss: 0.092
[06:56:37] Epoch 43 [>        ] loss: 0.085
[06:57:03] Epoch 43 [=>       ] loss: 0.085
[06:57:29] Epoch 43 [==>      ] loss: 0.069
[06:57:55] Epoch 43 [===>     ] loss: 0.082
[06:58:22] Epoch 43 [====>    ] loss: 0.093
[06:58:48] Epoch 43 [=====>   ] loss: 0.088
[06:59:14] Epoch 43 [======>  ] loss: 0.099
[06:59:40] Epoch 43 [=======> ] loss: 0.103
[07:00:06] Epoch 43 [========>] loss: 0.100
Test accuracy of the cnn on the 50000 train images: 99.70%
Test accuracy of the cnn on the 10000 test images: 77.31%
[07:02:24] Epoch 44 [         ] loss: 0.077
[07:02:50] Epoch 44 [>        ] loss: 0.082
[07:03:16] Epoch 44 [=>       ] loss: 0.088
[07:03:42] Epoch 44 [==>      ] loss: 0.082
[07:04:07] Epoch 44 [===>     ] loss: 0.093
[07:04:33] Epoch 44 [====>    ] loss: 0.094
[07:04:59] Epoch 44 [=====>   ] loss: 0.087
[07:05:25] Epoch 44 [======>  ] loss: 0.091
[07:05:52] Epoch 44 [=======> ] loss: 0.099
[07:06:18] Epoch 44 [========>] loss: 0.090
Test accuracy of the cnn on the 50000 train images: 99.86%
Test accuracy of the cnn on the 10000 test images: 77.48%
[07:08:35] Epoch 45 [         ] loss: 0.079
[07:09:01] Epoch 45 [>        ] loss: 0.075
[07:09:27] Epoch 45 [=>       ] loss: 0.080
[07:09:53] Epoch 45 [==>      ] loss: 0.081
[07:10:19] Epoch 45 [===>     ] loss: 0.095
[07:10:45] Epoch 45 [====>    ] loss: 0.097
[07:11:11] Epoch 45 [=====>   ] loss: 0.098
[07:11:37] Epoch 45 [======>  ] loss: 0.091
[07:12:03] Epoch 45 [=======> ] loss: 0.098
[07:12:29] Epoch 45 [========>] loss: 0.089
Test accuracy of the cnn on the 50000 train images: 99.85%
Test accuracy of the cnn on the 10000 test images: 78.22%
[07:14:48] Epoch 46 [         ] loss: 0.075
[07:15:14] Epoch 46 [>        ] loss: 0.086
[07:15:41] Epoch 46 [=>       ] loss: 0.076
[07:16:06] Epoch 46 [==>      ] loss: 0.083
[07:16:32] Epoch 46 [===>     ] loss: 0.079
[07:16:59] Epoch 46 [====>    ] loss: 0.087
[07:17:25] Epoch 46 [=====>   ] loss: 0.074
[07:17:51] Epoch 46 [======>  ] loss: 0.082
[07:18:17] Epoch 46 [=======> ] loss: 0.095
[07:18:43] Epoch 46 [========>] loss: 0.097
Test accuracy of the cnn on the 50000 train images: 99.84%
Test accuracy of the cnn on the 10000 test images: 77.58%
[07:21:01] Epoch 47 [         ] loss: 0.086
[07:21:27] Epoch 47 [>        ] loss: 0.093
[07:21:53] Epoch 47 [=>       ] loss: 0.070
[07:22:19] Epoch 47 [==>      ] loss: 0.069
[07:22:45] Epoch 47 [===>     ] loss: 0.078
[07:23:11] Epoch 47 [====>    ] loss: 0.092
[07:23:38] Epoch 47 [=====>   ] loss: 0.096
[07:24:04] Epoch 47 [======>  ] loss: 0.075
[07:24:30] Epoch 47 [=======> ] loss: 0.078
[07:24:56] Epoch 47 [========>] loss: 0.107
Test accuracy of the cnn on the 50000 train images: 99.83%
Test accuracy of the cnn on the 10000 test images: 77.98%
[07:27:13] Epoch 48 [         ] loss: 0.090
[07:27:39] Epoch 48 [>        ] loss: 0.071
[07:28:05] Epoch 48 [=>       ] loss: 0.080
[07:28:31] Epoch 48 [==>      ] loss: 0.088
[07:28:57] Epoch 48 [===>     ] loss: 0.094
[07:29:23] Epoch 48 [====>    ] loss: 0.087
[07:29:49] Epoch 48 [=====>   ] loss: 0.074
[07:30:15] Epoch 48 [======>  ] loss: 0.085
[07:30:41] Epoch 48 [=======> ] loss: 0.097
[07:31:07] Epoch 48 [========>] loss: 0.094
Test accuracy of the cnn on the 50000 train images: 99.80%
Test accuracy of the cnn on the 10000 test images: 77.54%
[07:33:24] Epoch 49 [         ] loss: 0.072
[07:33:50] Epoch 49 [>        ] loss: 0.078
[07:34:16] Epoch 49 [=>       ] loss: 0.063
[07:34:42] Epoch 49 [==>      ] loss: 0.074
[07:35:08] Epoch 49 [===>     ] loss: 0.095
[07:35:34] Epoch 49 [====>    ] loss: 0.075
[07:36:00] Epoch 49 [=====>   ] loss: 0.089
[07:36:26] Epoch 49 [======>  ] loss: 0.075
[07:36:52] Epoch 49 [=======> ] loss: 0.104
[07:37:18] Epoch 49 [========>] loss: 0.094
Test accuracy of the cnn on the 50000 train images: 99.74%
Test accuracy of the cnn on the 10000 test images: 76.88%
[07:39:37] Epoch 50 [         ] loss: 0.088
[07:40:04] Epoch 50 [>        ] loss: 0.086
[07:40:30] Epoch 50 [=>       ] loss: 0.085
[07:40:56] Epoch 50 [==>      ] loss: 0.091
[07:41:23] Epoch 50 [===>     ] loss: 0.083
[07:41:49] Epoch 50 [====>    ] loss: 0.082
[07:42:15] Epoch 50 [=====>   ] loss: 0.094
[07:42:41] Epoch 50 [======>  ] loss: 0.082
[07:43:07] Epoch 50 [=======> ] loss: 0.092
[07:43:33] Epoch 50 [========>] loss: 0.101
Test accuracy of the cnn on the 50000 train images: 99.73%
Test accuracy of the cnn on the 10000 test images: 76.88%
Finished Training
Results generated with seed 874450

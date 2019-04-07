Using seed 874450
CNN(
  (encoder): Sequential(
    (0): ConvLayer(
      (0): Conv2d(3, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): ReLU()
    )
    (1): ConvLayer(
      (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
      (1): ReLU()
      (2): Dropout(p=0.2)
    )
    (2): ConvLayer(
      (0): Conv2d(96, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): ReLU()
    )
    (3): ConvLayer(
      (0): Conv2d(192, 192, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
      (1): ReLU()
      (2): Dropout(p=0.5)
    )
  )
  (decoder): Sequential(
    (0): BatchNorm1d(12288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (1): FcLayer(
      (0): Linear(in_features=12288, out_features=256, bias=True)
      (1): ReLU()
    )
    (2): FcLayer(
      (0): Linear(in_features=256, out_features=10, bias=True)
    )
  )
)
Using 3,756,906 parameters
[('encoder.0.0.weight', 2592), ('encoder.0.0.bias', 96), ('encoder.1.0.weight', 82944), ('encoder.1.0.bias', 96), ('encoder.2.0.weight', 165888), ('encoder.2.0.bias', 192), ('encoder.3.0.weight', 331776), ('encoder.3.0.bias', 192), ('decoder.0.weight', 12288), ('decoder.0.bias', 12288), ('decoder.1.0.weight', 3145728), ('decoder.1.0.bias', 256), ('decoder.2.0.weight', 2560), ('decoder.2.0.bias', 10)]
TRAINING
==============================
Epoches: 50
Batch size: 128
Learning rate: 0.001

[07:46:03] Epoch 1 [         ] loss: 1.863
[07:46:31] Epoch 1 [>        ] loss: 1.679
[07:46:59] Epoch 1 [=>       ] loss: 1.513
[07:47:26] Epoch 1 [==>      ] loss: 1.414
[07:47:54] Epoch 1 [===>     ] loss: 1.358
[07:48:21] Epoch 1 [====>    ] loss: 1.336
[07:48:49] Epoch 1 [=====>   ] loss: 1.286
[07:49:16] Epoch 1 [======>  ] loss: 1.250
[07:49:44] Epoch 1 [=======> ] loss: 1.208
[07:50:12] Epoch 1 [========>] loss: 1.168
Test accuracy of the cnn on the 50000 train images: 61.23%
Test accuracy of the cnn on the 10000 test images: 59.30%
[07:53:45] Epoch 2 [         ] loss: 1.099
[07:54:14] Epoch 2 [>        ] loss: 1.091
[07:54:44] Epoch 2 [=>       ] loss: 1.061
[07:55:13] Epoch 2 [==>      ] loss: 1.058
[07:55:43] Epoch 2 [===>     ] loss: 1.066
[07:56:13] Epoch 2 [====>    ] loss: 1.043
[07:56:43] Epoch 2 [=====>   ] loss: 0.981
[07:57:13] Epoch 2 [======>  ] loss: 0.972
[07:57:44] Epoch 2 [=======> ] loss: 0.984
[07:58:15] Epoch 2 [========>] loss: 0.967
Test accuracy of the cnn on the 50000 train images: 67.50%
Test accuracy of the cnn on the 10000 test images: 64.10%
[08:01:54] Epoch 3 [         ] loss: 0.851
[08:02:24] Epoch 3 [>        ] loss: 0.863
[08:02:55] Epoch 3 [=>       ] loss: 0.862
[08:03:25] Epoch 3 [==>      ] loss: 0.851
[08:03:55] Epoch 3 [===>     ] loss: 0.852
[08:04:26] Epoch 3 [====>    ] loss: 0.852
[08:04:56] Epoch 3 [=====>   ] loss: 0.837
[08:05:26] Epoch 3 [======>  ] loss: 0.897
[08:05:57] Epoch 3 [=======> ] loss: 0.829
[08:06:27] Epoch 3 [========>] loss: 0.800
Test accuracy of the cnn on the 50000 train images: 78.26%
Test accuracy of the cnn on the 10000 test images: 72.48%
[08:10:11] Epoch 4 [         ] loss: 0.726
[08:10:42] Epoch 4 [>        ] loss: 0.717
[08:11:13] Epoch 4 [=>       ] loss: 0.751
[08:11:44] Epoch 4 [==>      ] loss: 0.716
[08:12:15] Epoch 4 [===>     ] loss: 0.702
[08:12:46] Epoch 4 [====>    ] loss: 0.723
[08:13:16] Epoch 4 [=====>   ] loss: 0.732
[08:13:47] Epoch 4 [======>  ] loss: 0.721
[08:14:18] Epoch 4 [=======> ] loss: 0.708
[08:14:49] Epoch 4 [========>] loss: 0.705
Test accuracy of the cnn on the 50000 train images: 81.50%
Test accuracy of the cnn on the 10000 test images: 73.70%
[08:18:30] Epoch 5 [         ] loss: 0.592
[08:19:01] Epoch 5 [>        ] loss: 0.581
[08:19:32] Epoch 5 [=>       ] loss: 0.593
[08:20:03] Epoch 5 [==>      ] loss: 0.606
[08:20:34] Epoch 5 [===>     ] loss: 0.637
[08:21:05] Epoch 5 [====>    ] loss: 0.635
[08:21:36] Epoch 5 [=====>   ] loss: 0.612
[08:22:06] Epoch 5 [======>  ] loss: 0.618
[08:22:37] Epoch 5 [=======> ] loss: 0.623
[08:23:07] Epoch 5 [========>] loss: 0.614
Test accuracy of the cnn on the 50000 train images: 86.28%
Test accuracy of the cnn on the 10000 test images: 76.60%
[08:26:50] Epoch 6 [         ] loss: 0.518
[08:27:20] Epoch 6 [>        ] loss: 0.515
[08:27:50] Epoch 6 [=>       ] loss: 0.504
[08:28:21] Epoch 6 [==>      ] loss: 0.537
[08:28:51] Epoch 6 [===>     ] loss: 0.530
[08:29:21] Epoch 6 [====>    ] loss: 0.529
[08:29:51] Epoch 6 [=====>   ] loss: 0.528
[08:30:22] Epoch 6 [======>  ] loss: 0.552
[08:30:52] Epoch 6 [=======> ] loss: 0.549
[08:31:23] Epoch 6 [========>] loss: 0.572
Test accuracy of the cnn on the 50000 train images: 87.31%
Test accuracy of the cnn on the 10000 test images: 76.60%
[08:35:05] Epoch 7 [         ] loss: 0.438
[08:35:35] Epoch 7 [>        ] loss: 0.436
[08:36:06] Epoch 7 [=>       ] loss: 0.476
[08:36:36] Epoch 7 [==>      ] loss: 0.463
[08:37:06] Epoch 7 [===>     ] loss: 0.458
[08:37:37] Epoch 7 [====>    ] loss: 0.471
[08:38:07] Epoch 7 [=====>   ] loss: 0.463
[08:38:37] Epoch 7 [======>  ] loss: 0.494
[08:39:07] Epoch 7 [=======> ] loss: 0.500
[08:39:38] Epoch 7 [========>] loss: 0.480
Test accuracy of the cnn on the 50000 train images: 90.45%
Test accuracy of the cnn on the 10000 test images: 77.05%
[08:43:21] Epoch 8 [         ] loss: 0.411
[08:43:52] Epoch 8 [>        ] loss: 0.388
[08:44:22] Epoch 8 [=>       ] loss: 0.409
[08:44:53] Epoch 8 [==>      ] loss: 0.394
[08:45:24] Epoch 8 [===>     ] loss: 0.440
[08:45:54] Epoch 8 [====>    ] loss: 0.399
[08:46:25] Epoch 8 [=====>   ] loss: 0.438
[08:46:56] Epoch 8 [======>  ] loss: 0.408
[08:47:26] Epoch 8 [=======> ] loss: 0.422
[08:47:57] Epoch 8 [========>] loss: 0.441
Test accuracy of the cnn on the 50000 train images: 94.08%
Test accuracy of the cnn on the 10000 test images: 78.82%
[08:51:39] Epoch 9 [         ] loss: 0.341
[08:52:10] Epoch 9 [>        ] loss: 0.340
[08:52:40] Epoch 9 [=>       ] loss: 0.345
[08:53:11] Epoch 9 [==>      ] loss: 0.362
[08:53:41] Epoch 9 [===>     ] loss: 0.365
[08:54:12] Epoch 9 [====>    ] loss: 0.359
[08:54:43] Epoch 9 [=====>   ] loss: 0.404
[08:55:13] Epoch 9 [======>  ] loss: 0.384
[08:55:44] Epoch 9 [=======> ] loss: 0.378
[08:56:15] Epoch 9 [========>] loss: 0.384
Test accuracy of the cnn on the 50000 train images: 95.71%
Test accuracy of the cnn on the 10000 test images: 78.92%
[08:59:58] Epoch 10 [         ] loss: 0.280
[09:00:28] Epoch 10 [>        ] loss: 0.299
[09:00:59] Epoch 10 [=>       ] loss: 0.324
[09:01:29] Epoch 10 [==>      ] loss: 0.298
[09:02:00] Epoch 10 [===>     ] loss: 0.323
[09:02:31] Epoch 10 [====>    ] loss: 0.309
[09:03:02] Epoch 10 [=====>   ] loss: 0.329
[09:03:32] Epoch 10 [======>  ] loss: 0.336
[09:04:04] Epoch 10 [=======> ] loss: 0.326
[09:04:34] Epoch 10 [========>] loss: 0.334
Test accuracy of the cnn on the 50000 train images: 96.35%
Test accuracy of the cnn on the 10000 test images: 79.16%
[09:08:16] Epoch 11 [         ] loss: 0.255
[09:08:46] Epoch 11 [>        ] loss: 0.258
[09:09:17] Epoch 11 [=>       ] loss: 0.263
[09:09:47] Epoch 11 [==>      ] loss: 0.272
[09:10:18] Epoch 11 [===>     ] loss: 0.282
[09:10:48] Epoch 11 [====>    ] loss: 0.289
[09:11:19] Epoch 11 [=====>   ] loss: 0.303
[09:11:49] Epoch 11 [======>  ] loss: 0.309
[09:12:20] Epoch 11 [=======> ] loss: 0.294
[09:12:50] Epoch 11 [========>] loss: 0.306
Test accuracy of the cnn on the 50000 train images: 97.70%
Test accuracy of the cnn on the 10000 test images: 79.22%
[09:16:32] Epoch 12 [         ] loss: 0.236
[09:17:03] Epoch 12 [>        ] loss: 0.236
[09:17:33] Epoch 12 [=>       ] loss: 0.250
[09:18:04] Epoch 12 [==>      ] loss: 0.250
[09:18:34] Epoch 12 [===>     ] loss: 0.252
[09:19:05] Epoch 12 [====>    ] loss: 0.257
[09:19:35] Epoch 12 [=====>   ] loss: 0.273
[09:20:05] Epoch 12 [======>  ] loss: 0.252
[09:20:35] Epoch 12 [=======> ] loss: 0.280
[09:21:06] Epoch 12 [========>] loss: 0.287
Test accuracy of the cnn on the 50000 train images: 97.90%
Test accuracy of the cnn on the 10000 test images: 79.14%
[09:24:47] Epoch 13 [         ] loss: 0.201
[09:25:18] Epoch 13 [>        ] loss: 0.208
[09:25:49] Epoch 13 [=>       ] loss: 0.218
[09:26:19] Epoch 13 [==>      ] loss: 0.220
[09:26:49] Epoch 13 [===>     ] loss: 0.244
[09:27:23] Epoch 13 [====>    ] loss: 0.267
[09:27:55] Epoch 13 [=====>   ] loss: 0.256
[09:28:28] Epoch 13 [======>  ] loss: 0.251
[09:29:00] Epoch 13 [=======> ] loss: 0.263
[09:29:32] Epoch 13 [========>] loss: 0.244
Test accuracy of the cnn on the 50000 train images: 98.27%
Test accuracy of the cnn on the 10000 test images: 79.22%
[09:33:17] Epoch 14 [         ] loss: 0.194
[09:33:47] Epoch 14 [>        ] loss: 0.194
[09:34:18] Epoch 14 [=>       ] loss: 0.192
[09:34:48] Epoch 14 [==>      ] loss: 0.215
[09:35:19] Epoch 14 [===>     ] loss: 0.214
[09:35:49] Epoch 14 [====>    ] loss: 0.223
[09:36:19] Epoch 14 [=====>   ] loss: 0.240
[09:36:50] Epoch 14 [======>  ] loss: 0.217
[09:37:20] Epoch 14 [=======> ] loss: 0.235
[09:37:50] Epoch 14 [========>] loss: 0.233
Test accuracy of the cnn on the 50000 train images: 98.46%
Test accuracy of the cnn on the 10000 test images: 79.41%
[09:41:31] Epoch 15 [         ] loss: 0.175
[09:42:02] Epoch 15 [>        ] loss: 0.188
[09:42:32] Epoch 15 [=>       ] loss: 0.182
[09:43:03] Epoch 15 [==>      ] loss: 0.204
[09:43:34] Epoch 15 [===>     ] loss: 0.180
[09:44:04] Epoch 15 [====>    ] loss: 0.205
[09:44:35] Epoch 15 [=====>   ] loss: 0.198
[09:45:06] Epoch 15 [======>  ] loss: 0.205
[09:45:36] Epoch 15 [=======> ] loss: 0.211
[09:46:07] Epoch 15 [========>] loss: 0.206
Test accuracy of the cnn on the 50000 train images: 98.92%
Test accuracy of the cnn on the 10000 test images: 80.10%
[09:49:50] Epoch 16 [         ] loss: 0.153
[09:50:21] Epoch 16 [>        ] loss: 0.170
[09:50:52] Epoch 16 [=>       ] loss: 0.185
[09:51:22] Epoch 16 [==>      ] loss: 0.195
[09:51:53] Epoch 16 [===>     ] loss: 0.186
[09:52:24] Epoch 16 [====>    ] loss: 0.183
[09:52:55] Epoch 16 [=====>   ] loss: 0.177
[09:53:25] Epoch 16 [======>  ] loss: 0.198
[09:53:56] Epoch 16 [=======> ] loss: 0.207
[09:54:26] Epoch 16 [========>] loss: 0.206
Test accuracy of the cnn on the 50000 train images: 98.73%
Test accuracy of the cnn on the 10000 test images: 79.57%
[09:58:08] Epoch 17 [         ] loss: 0.146
[09:58:38] Epoch 17 [>        ] loss: 0.163
[09:59:09] Epoch 17 [=>       ] loss: 0.168
[09:59:40] Epoch 17 [==>      ] loss: 0.153
[10:00:11] Epoch 17 [===>     ] loss: 0.157
[10:00:41] Epoch 17 [====>    ] loss: 0.163
[10:01:12] Epoch 17 [=====>   ] loss: 0.174
[10:01:42] Epoch 17 [======>  ] loss: 0.211
[10:02:13] Epoch 17 [=======> ] loss: 0.191
[10:02:43] Epoch 17 [========>] loss: 0.192
Test accuracy of the cnn on the 50000 train images: 99.12%
Test accuracy of the cnn on the 10000 test images: 79.20%
[10:06:26] Epoch 18 [         ] loss: 0.155
[10:06:57] Epoch 18 [>        ] loss: 0.171
[10:07:27] Epoch 18 [=>       ] loss: 0.171
[10:07:57] Epoch 18 [==>      ] loss: 0.159
[10:08:27] Epoch 18 [===>     ] loss: 0.170
[10:08:57] Epoch 18 [====>    ] loss: 0.166
[10:09:28] Epoch 18 [=====>   ] loss: 0.175
[10:09:58] Epoch 18 [======>  ] loss: 0.172
[10:10:28] Epoch 18 [=======> ] loss: 0.193
[10:10:58] Epoch 18 [========>] loss: 0.178
Test accuracy of the cnn on the 50000 train images: 99.02%
Test accuracy of the cnn on the 10000 test images: 79.35%
[10:14:39] Epoch 19 [         ] loss: 0.146
[10:15:09] Epoch 19 [>        ] loss: 0.146
[10:15:39] Epoch 19 [=>       ] loss: 0.141
[10:16:10] Epoch 19 [==>      ] loss: 0.138
[10:16:40] Epoch 19 [===>     ] loss: 0.151
[10:17:11] Epoch 19 [====>    ] loss: 0.162
[10:17:41] Epoch 19 [=====>   ] loss: 0.166
[10:18:12] Epoch 19 [======>  ] loss: 0.175
[10:18:42] Epoch 19 [=======> ] loss: 0.162
[10:19:13] Epoch 19 [========>] loss: 0.167
Test accuracy of the cnn on the 50000 train images: 99.37%
Test accuracy of the cnn on the 10000 test images: 79.97%
[10:22:55] Epoch 20 [         ] loss: 0.119
[10:23:25] Epoch 20 [>        ] loss: 0.139
[10:23:55] Epoch 20 [=>       ] loss: 0.135
[10:24:25] Epoch 20 [==>      ] loss: 0.145
[10:24:56] Epoch 20 [===>     ] loss: 0.137
[10:25:26] Epoch 20 [====>    ] loss: 0.147
[10:25:57] Epoch 20 [=====>   ] loss: 0.144
[10:26:27] Epoch 20 [======>  ] loss: 0.139
[10:26:57] Epoch 20 [=======> ] loss: 0.160
[10:27:28] Epoch 20 [========>] loss: 0.140
Test accuracy of the cnn on the 50000 train images: 99.31%
Test accuracy of the cnn on the 10000 test images: 79.70%
[10:31:08] Epoch 21 [         ] loss: 0.137
[10:31:39] Epoch 21 [>        ] loss: 0.136
[10:32:10] Epoch 21 [=>       ] loss: 0.125
[10:32:40] Epoch 21 [==>      ] loss: 0.119
[10:33:11] Epoch 21 [===>     ] loss: 0.127
[10:33:42] Epoch 21 [====>    ] loss: 0.142
[10:34:12] Epoch 21 [=====>   ] loss: 0.154
[10:34:43] Epoch 21 [======>  ] loss: 0.161
[10:35:13] Epoch 21 [=======> ] loss: 0.158
[10:35:44] Epoch 21 [========>] loss: 0.158
Test accuracy of the cnn on the 50000 train images: 99.45%
Test accuracy of the cnn on the 10000 test images: 79.54%
[10:39:27] Epoch 22 [         ] loss: 0.133
[10:39:58] Epoch 22 [>        ] loss: 0.110
[10:40:28] Epoch 22 [=>       ] loss: 0.124
[10:40:59] Epoch 22 [==>      ] loss: 0.118
[10:41:29] Epoch 22 [===>     ] loss: 0.135
[10:42:00] Epoch 22 [====>    ] loss: 0.148
[10:42:31] Epoch 22 [=====>   ] loss: 0.141
[10:43:01] Epoch 22 [======>  ] loss: 0.156
[10:43:32] Epoch 22 [=======> ] loss: 0.159
[10:44:03] Epoch 22 [========>] loss: 0.156
Test accuracy of the cnn on the 50000 train images: 99.62%
Test accuracy of the cnn on the 10000 test images: 79.55%
[10:47:44] Epoch 23 [         ] loss: 0.116
[10:48:15] Epoch 23 [>        ] loss: 0.106
[10:48:45] Epoch 23 [=>       ] loss: 0.132
[10:49:17] Epoch 23 [==>      ] loss: 0.138
[10:49:48] Epoch 23 [===>     ] loss: 0.122
[10:50:18] Epoch 23 [====>    ] loss: 0.134
[10:50:49] Epoch 23 [=====>   ] loss: 0.146
[10:51:20] Epoch 23 [======>  ] loss: 0.138
[10:51:50] Epoch 23 [=======> ] loss: 0.141
[10:52:21] Epoch 23 [========>] loss: 0.152
Test accuracy of the cnn on the 50000 train images: 99.49%
Test accuracy of the cnn on the 10000 test images: 79.58%
[10:56:02] Epoch 24 [         ] loss: 0.128
[10:56:33] Epoch 24 [>        ] loss: 0.125
[10:57:03] Epoch 24 [=>       ] loss: 0.118
[10:57:33] Epoch 24 [==>      ] loss: 0.129
[10:58:04] Epoch 24 [===>     ] loss: 0.147
[10:58:34] Epoch 24 [====>    ] loss: 0.128
[10:59:04] Epoch 24 [=====>   ] loss: 0.118
[10:59:35] Epoch 24 [======>  ] loss: 0.129
[11:00:06] Epoch 24 [=======> ] loss: 0.140
[11:00:37] Epoch 24 [========>] loss: 0.132
Test accuracy of the cnn on the 50000 train images: 99.62%
Test accuracy of the cnn on the 10000 test images: 80.07%
[11:04:18] Epoch 25 [         ] loss: 0.115
[11:04:48] Epoch 25 [>        ] loss: 0.108
[11:05:19] Epoch 25 [=>       ] loss: 0.117
[11:05:49] Epoch 25 [==>      ] loss: 0.116
[11:06:20] Epoch 25 [===>     ] loss: 0.113
[11:06:51] Epoch 25 [====>    ] loss: 0.129
[11:07:21] Epoch 25 [=====>   ] loss: 0.122
[11:07:52] Epoch 25 [======>  ] loss: 0.124
[11:08:22] Epoch 25 [=======> ] loss: 0.122
[11:08:53] Epoch 25 [========>] loss: 0.129
Test accuracy of the cnn on the 50000 train images: 99.54%
Test accuracy of the cnn on the 10000 test images: 79.64%
[11:12:34] Epoch 26 [         ] loss: 0.109
[11:13:05] Epoch 26 [>        ] loss: 0.121
[11:13:36] Epoch 26 [=>       ] loss: 0.112
[11:14:06] Epoch 26 [==>      ] loss: 0.131
[11:14:37] Epoch 26 [===>     ] loss: 0.107
[11:15:08] Epoch 26 [====>    ] loss: 0.136
[11:15:38] Epoch 26 [=====>   ] loss: 0.138
[11:16:09] Epoch 26 [======>  ] loss: 0.135
[11:16:40] Epoch 26 [=======> ] loss: 0.138
[11:17:10] Epoch 26 [========>] loss: 0.157
Test accuracy of the cnn on the 50000 train images: 99.68%
Test accuracy of the cnn on the 10000 test images: 79.84%
[11:20:52] Epoch 27 [         ] loss: 0.113
[11:21:23] Epoch 27 [>        ] loss: 0.117
[11:21:54] Epoch 27 [=>       ] loss: 0.112
[11:22:24] Epoch 27 [==>      ] loss: 0.104
[11:22:55] Epoch 27 [===>     ] loss: 0.104
[11:23:25] Epoch 27 [====>    ] loss: 0.126
[11:23:55] Epoch 27 [=====>   ] loss: 0.111
[11:24:25] Epoch 27 [======>  ] loss: 0.125
[11:24:55] Epoch 27 [=======> ] loss: 0.131
[11:25:24] Epoch 27 [========>] loss: 0.131
Test accuracy of the cnn on the 50000 train images: 99.71%
Test accuracy of the cnn on the 10000 test images: 79.32%
[11:29:01] Epoch 28 [         ] loss: 0.109
[11:29:31] Epoch 28 [>        ] loss: 0.104
[11:30:02] Epoch 28 [=>       ] loss: 0.093
[11:30:32] Epoch 28 [==>      ] loss: 0.118
[11:31:02] Epoch 28 [===>     ] loss: 0.110
[11:31:33] Epoch 28 [====>    ] loss: 0.101
[11:32:03] Epoch 28 [=====>   ] loss: 0.115
[11:32:34] Epoch 28 [======>  ] loss: 0.113
[11:33:04] Epoch 28 [=======> ] loss: 0.116
[11:33:35] Epoch 28 [========>] loss: 0.118
Test accuracy of the cnn on the 50000 train images: 99.78%
Test accuracy of the cnn on the 10000 test images: 79.87%
[11:37:12] Epoch 29 [         ] loss: 0.104
[11:37:43] Epoch 29 [>        ] loss: 0.101
[11:38:13] Epoch 29 [=>       ] loss: 0.101
[11:38:44] Epoch 29 [==>      ] loss: 0.107
[11:39:15] Epoch 29 [===>     ] loss: 0.111
[11:39:45] Epoch 29 [====>    ] loss: 0.114
[11:40:16] Epoch 29 [=====>   ] loss: 0.104
[11:40:46] Epoch 29 [======>  ] loss: 0.126
[11:41:16] Epoch 29 [=======> ] loss: 0.112
[11:41:46] Epoch 29 [========>] loss: 0.118
Test accuracy of the cnn on the 50000 train images: 99.78%
Test accuracy of the cnn on the 10000 test images: 79.71%
[11:45:24] Epoch 30 [         ] loss: 0.092
[11:45:54] Epoch 30 [>        ] loss: 0.108
[11:46:26] Epoch 30 [=>       ] loss: 0.109
[11:46:57] Epoch 30 [==>      ] loss: 0.108
[11:47:30] Epoch 30 [===>     ] loss: 0.103
[11:48:03] Epoch 30 [====>    ] loss: 0.112
[11:48:36] Epoch 30 [=====>   ] loss: 0.118
[11:49:08] Epoch 30 [======>  ] loss: 0.117
[11:49:39] Epoch 30 [=======> ] loss: 0.119
[11:50:11] Epoch 30 [========>] loss: 0.130
Test accuracy of the cnn on the 50000 train images: 99.65%
Test accuracy of the cnn on the 10000 test images: 79.32%
[11:53:45] Epoch 31 [         ] loss: 0.097
[11:54:15] Epoch 31 [>        ] loss: 0.101
[11:54:44] Epoch 31 [=>       ] loss: 0.105
[11:55:13] Epoch 31 [==>      ] loss: 0.104
[11:55:42] Epoch 31 [===>     ] loss: 0.121
[11:56:12] Epoch 31 [====>    ] loss: 0.101
[11:56:41] Epoch 31 [=====>   ] loss: 0.107
[11:57:11] Epoch 31 [======>  ] loss: 0.109
[11:57:41] Epoch 31 [=======> ] loss: 0.108
[11:58:12] Epoch 31 [========>] loss: 0.109

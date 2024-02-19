# 映像特徴量を用いたサッカー実況文章の生成

# 実行環境

学科サーバー(amane)を用いる。2024/02/14時点で動作確認済み。


# 手順

## ベースとなるイメージの用意

以下を実行する。
```
singularity pull docker://huggingface/transformers-pytorch-gpu:4.35.2
singularity pull docker://nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
```

それぞれ`I3Dからの特徴量抽出`と`BARTのファインチューニング`の環境構築時に使用するため、パスを覚えておく。

## 映像の用意

YouTubeからの映像の取得にyt-dlpを用いる。`yt-dlp -f 18 リンク`を実行しダウンロードを行う。


## Whisper実行

Whisper/を参照。


## I3Dからの特徴量抽出

I3D/を参照。


## BARTのファインチューニング

BART/を参照。
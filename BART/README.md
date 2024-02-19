1. `env_train.def`の2行目の`/PATH/TO/transformers-pytorch-gpu_4.35.2.sif`を実際のパスに書き換えてビルドをする。
```
singularity build --fakeroot env_train.sif env_train.def
```

2. ex1〜ex3のsbatchファイルの引数を書き換えて実行する。

ex1、ex2 の引数は以下の通り。
```
singularity exec --nv env_train.sif python3 ex1.py  \
							何試合分か  \
							データセットディレクトリへのパス  \
							結果出力先  \
							何epoch \
                            学習率
```

ex3は先に`quantify.py`を実行してI3D特徴量を量子化トークンに変換してから行う。
ファインチューニング時にex3.py内部のパスを書き換える。
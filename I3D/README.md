1. 以下を実行。
```
git clone https://github.com/v-iashin/video_features.git
cd video_features
cp i3d-extractor.def extract.sbatch extract_i3d.py ./
mkdir logs
```

1. `i3d-extractor.def`を編集する。2行目の`/PATH/TO/cuda_11.8.0-cudnn8-devel-ubuntu22.04.sif`のパスを実際のパスに書き換える。

2. 以下を実行する。
```
singularity build --fakeroot i3d-extractor-ubt22.sif i3d-extractor.def
```

3. `extract.sbatch`の引数を書き換えて`sbatch extract.sbatch`を実行することで、I3Dからの特徴量抽出を行う。

`extract_i3d.py`の引数は
- Whisperで書き起こした発話&タイムスタンプのcsvファイルへのパス
    - I3D実行途中のjoblibファイル指定も可能
- 試合映像ファイルのパス
- 結果保存先ディレクトリ
- タイムスタンプ区切りの動画の一時保存先
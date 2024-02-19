import joblib
import glob
import pandas as pd
import numpy as np

def concat_df(num_videos, data_dir_path):
    # joblibファイルの読み込みと結合を行う
    files = glob.glob( data_dir_path + "*")
    files.sort()

    df = joblib.load(files[0])
    df = df.drop("Unnamed: 0", axis=1)  # 発話idをdrop
    df["video_id"] = 1  # 何番目のvideoかの情報を追加

    for i in range(1, num_videos):
        tmp_df = joblib.load(files[i])
        tmp_df = tmp_df.drop("Unnamed: 0", axis=1)
        tmp_df["video_id"] = i+1

        df = pd.concat([df, tmp_df], axis=0)
    df = df.reset_index(drop=True)
    return df

data_dir_path =  "../i3d_datasets/"  ## データセットディレクトリへのパス
df = concat_df(26, data_dir_path)

from sklearn.decomposition import PCA

def dimensionality_reduction(all_data, cut_dimension_size):
    # PCAしてcut_dimmension_size次元にする

    ## (データ数, hoge, 1024)次元→(データ数xhoge, 1024)次元にする
    all_data_vecs = []
    stack_dimension = []  ## 復元用
    for data in all_data:
        for vecs in data:
            all_data_vecs.append(vecs)
        stack_dimension.append(len(data))

    pca = PCA(n_components=cut_dimension_size)
    transformed_data = pca.fit_transform(all_data_vecs)

    ## (データ数xhoge, 1024)次元→(データ数, hoge, 1024)次元へ戻す
    return_vecs = []
    transformed_data = transformed_data.tolist()
    for dim in stack_dimension:
        tmp = []
        for _ in range(dim):
            tmp.append(transformed_data.pop(0))
        return_vecs.append(tmp)
    return return_vecs

def quantize_and_map(df, column_name, cut_dimension_size, quantify_size):
    all_data = []

    for _vectors in df["i3d_feature"]:
        all_data.append(_vectors.tolist())
    all_data = dimensionality_reduction(all_data, cut_dimension_size)  ## PCAでcut_dimension_size次元に削減

    # ネストされたリストをフラット化する関数
    def flatten(lst):
        for elem in lst:
            if isinstance(elem, list):
                yield from flatten(elem)
            else:
                yield elem

    # リストをフラット化
    flat_list = list(flatten(all_data))

    # 全ベクトルの最小値と最大値を求める
    min_value = min(flat_list)
    max_value = max(flat_list)

    # 量子化のための範囲とステップを計算
    ranges = np.linspace(min_value, max_value, num=quantify_size)

    tokens = [f"<i3d_{i}>" for i in range(quantify_size)]

    # 各ベクトルの各次元を量子化し、トークンにマッピング
    quantized_vectors = []
    for vectors in all_data:
        temp = []
        for vec in vectors:
            quantized_vec = []
            for val in vec:
                # どの範囲に値があるかを見つけ、対応するトークンを割り当てる
                for i in range(len(ranges)-1):
                    if ranges[i] <= val < ranges[i+1]:
                        quantized_vec.append(f"<i3d_{i+1}>")
                        break
                # 最大値の場合の処理
                if val >= max_value:
                    quantized_vec.append(f"<i3d_{quantify_size}>")
            temp.append(quantized_vec)
        quantized_vectors.append(temp)

    return tokens, quantized_vectors

tokens, new_vectors = quantize_and_map(df, "i3d_feature", 33, 2**5)
df["i3d_feature_quantiy"] = new_vectors
joblib.dump(df, "./all_video_i3d_quantify.joblib")
with open("./tokens.txt", "w") as f:
    for token in tokens:
        f.write(token + "\n")
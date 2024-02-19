import sys
import ffmpeg
from models.i3d.extract_i3d import ExtractI3D
from utils.utils import build_cfg_path
from omegaconf import OmegaConf
import numpy as np
import joblib
import pandas as pd
import datetime

def adjustment_video_len(start, end, base=10):
    if (end-start) < base:
        end = start+base
    return end-10, end+1

def clip_video(out_dir, video_path, start, end):
    output_file_name = "tmp.mp4"
    start = np.float64(start)
    end = np.float64(end)
    v_start, v_end = adjustment_video_len(start, end)
    
    stream = ffmpeg.input(video_path).trim(start=v_start, end=v_end).setpts('PTS-STARTPTS').output(out_dir+output_file_name)
    ffmpeg.run(stream, overwrite_output=True, quiet=True)

def extract_video_feature(df_path, video_path, out_path, tmp_dir):
    if ".joblib" in df_path:
        speeking_text_df = joblib.load(df_path)
    else:
        speeking_text_df = pd.read_csv(df_path)
        speeking_text_df = speeking_text_df.assign(i3d_feature=None)

    feature_type = "i3d"
    # Load and patch the config
    args = OmegaConf.load(build_cfg_path(feature_type))
    args.flow_type = "raft"
    args.stack_size = 25 # default 64
    # Load the model
    extractor = ExtractI3D(args)

    for index, item in speeking_text_df.iterrows():
        if item["i3d_feature"] is None:
            # 映像の対象範囲を切り取り保存
            clip_video(tmp_dir, video_path, item["start"], item["end"])

            # i3dを実行して特徴量を抽出
            clipped_video_path = tmp_dir+"tmp.mp4"
            feature_dict = extractor.extract(clipped_video_path)

            # dfに結果を格納
            speeking_text_df.at[index, "i3d_feature"] = feature_dict["rgb"]
            # 保存
            joblib.dump(speeking_text_df, out_path+"result_i3d.joblib")
            with open(tmp_dir+"log-end-index.txt", "a") as file:
                file.write(f"{datetime.datetime.now()} done: {index}\n")


def main():
    args = sys.argv
    df_path = args[1]
    video_path = args[2]
    out_path = args[3]
    tmp_dir = args[4]

    extract_video_feature(df_path, video_path, out_path, tmp_dir)

if __name__ == "__main__":
    main()

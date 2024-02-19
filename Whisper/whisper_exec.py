import whisper
import pandas as pd

def extract_speak(video_path, model):
    audio = whisper.load_audio(video_path)
    result = model.transcribe(audio, verbose=True, language="en")
    return result

def run(video_path, output_dir, output_file_name):
    model = whisper.load_model("small") # tiny→base→small(デフォルト)→midium→large
    result = extract_speak(video_path, model)
    df = pd.DataFrame.from_dict(result["segments"])
    df = df[["start", "end", "text"]]
    df.to_csv(output_dir + output_file_name + ".csv")

# それぞれpathを指定して実行
video_path = "PATH/TO/video.mp4"
output_dir = "PATH/TO/OUTPUT"
output_file_name = "video_result"
run(video_path, output_dir, output_file_name)

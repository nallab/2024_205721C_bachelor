import glob
import os
import sys
import pandas as pd
import joblib
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
import numpy as np
import evaluate
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import EarlyStoppingCallback
import json
from transformers import PreTrainedTokenizerBase
from dataclasses import dataclass
from typing import Optional, Any, Union
from transformers.tokenization_utils_base import PaddingStrategy
import torch.utils.data as data_utils


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


def preprocessing(df):
    ## → リスト化するだけ
    max_length = 50  # データ長をmax_lengthに合わせる
    num_data = len(df)
    features = []
    for i in range(num_data):
        # dfの各要素のndarrayをtorch.tensor型に変換
        i3d_f = torch.from_numpy(df["i3d_feature"][i].astype(np.float32)).clone()

        ## 一旦異常値データを無視する(あとで修正
        if max_length-i3d_f.shape[0] < 0:
            continue

        # textを格納
        label_text = df["text"][i]

        # リストに格納
        features.append([i3d_f, label_text])
        ## --> [[ I3D結果, label], ...]
    return features


# Data Collatorを用意
@dataclass
class DataCollatorForMe:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    max_length: Optional[int] = 50
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        new_features = {}
        i3d_features = None
        attention_mask = None
        labels = None

        self.model.to("cpu")

        # ここで辞書のリストを用意
        for feature in features:
            with torch.no_grad():
                i3d_f = feature[0].clone().detach()
                bos_embeds = self.model.model.encoder.embed_tokens(torch.tensor([self.tokenizer.bos_token_id]))
                eos_embeds = self.model.model.encoder.embed_tokens(torch.tensor([self.tokenizer.eos_token_id]))
                pad_embed = self.model.model.encoder.embed_tokens(torch.tensor([self.tokenizer.pad_token_id]))
                special_token_embed = self.model.model.encoder.embed_tokens(self.tokenizer.encode("<i3d>", add_special_tokens=False, return_tensors="pt")[0])  ## 特殊トークンベクトルを用意
                i3d_f = torch.cat((special_token_embed, i3d_f), dim=0)  ## 特殊トークンベクトル付与
                i3d_f = torch.cat((bos_embeds, i3d_f), dim=0)
                i3d_f = torch.cat((i3d_f, eos_embeds), dim=0)

                i3d_attention_mask = [1 if j < i3d_f.shape[0] else 0 for j in range(self.max_length)]
                i3d_attention_mask = torch.tensor(i3d_attention_mask)
                i3d_attention_mask = torch.reshape(i3d_attention_mask, (1, -1))

                concat_embed = pad_embed.repeat((self.max_length-i3d_f.shape[0], 1))
                i3d_f = torch.cat((i3d_f ,concat_embed), dim=0)
                i3d_f = torch.reshape(i3d_f, (1, self.max_length, 1024))

                text = feature[1]
                label = self.tokenizer.encode(text, add_special_tokens=True, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
                label = torch.where(label != 1, label, -100)  ## labelのpaddingは-100
                if i3d_features is None:
                    i3d_features = i3d_f
                    attention_mask = i3d_attention_mask
                    labels = label
                else:
                    i3d_features = torch.cat((i3d_features, i3d_f), dim=0)
                    attention_mask = torch.cat((attention_mask, i3d_attention_mask), dim=0)
                    labels = torch.cat((labels, label), dim=0)
                

        new_features["inputs_embeds"] = i3d_features
        new_features["attention_mask"] = attention_mask
        new_features["labels"] = labels

        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=new_features["labels"])
            new_features["decoder_input_ids"] = decoder_input_ids

        self.model.to("cuda:0")

        return new_features


# BLEUを利用するための関数を用意
metric = evaluate.load("bleu")
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    return preds, labels
def compute_metrics_bleu(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    # Replace -100s used for padding as we can't decode them
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["bleu"]}

    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
    ]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

args = sys.argv
num_videos = int(args[1])    # 試合数
data_dir_path = args[2]      # データセットのディレクトリへのパス
output_result_dir = args[3]  # 結果出力先
num_epochs = int(args[4])    # 何epoch実行するか
lr = float(args[5])          # 学習率, default: 5e-5

tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")

# 特殊トークン追加
tokenizer.add_special_tokens({"additional_special_tokens" : ["<i3d>"]})
# print("vocab size  (after): ", len(tokenizer))
model.resize_token_embeddings(len(tokenizer))
# 新しいembedding層を取得
new_embedding_layer = model.get_input_embeddings()
# padding_idxを再設定 (通常は0または特定のインデックス)
new_embedding_layer.padding_idx = tokenizer.pad_token_id  ## 1

df = concat_df(num_videos, data_dir_path)      # datasetの読み込み
dataset = preprocessing(df)  # 前処理

# データセットを分割
generator = torch.Generator().manual_seed(42)  # シード固定(同じ結果になるようにする)
tr_data, ev_data, ts_data = torch.utils.data.random_split(dataset, [0.7, 0.2, 0.1], generator=generator)

# data collatorを用意
data_collator = DataCollatorForMe(tokenizer, model)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = model.to(device)


# 学習条件
training_args = Seq2SeqTrainingArguments(
    output_dir = output_result_dir,
    predict_with_generate = True,
    per_device_train_batch_size = 16,
    per_device_eval_batch_size = 16,
    save_strategy = "epoch",
    save_total_limit = 2,
    evaluation_strategy = "epoch",
    load_best_model_at_end = True,
    logging_strategy = "epoch",
    learning_rate = lr,  # default: 5e-5
    weight_decay = 0,  # default: 0
    warmup_steps = 0,  # default: 0
    do_train = True,
    do_eval = True,
    num_train_epochs=num_epochs,
)

trainer = Seq2SeqTrainer(
    model= model,
    args=training_args,
    compute_metrics=compute_metrics_bleu,
    train_dataset=tr_data,
    eval_dataset=ev_data,
    data_collator=data_collator 
    # callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],  ## early_stopping
)

trainer.evaluate()
trainer.train()

tokenizer.save_pretrained(output_result_dir)
model.save_pretrained(output_result_dir)

# 10種類文章生成させてみる(open test)
with open(output_result_dir+"open_test.txt", "w") as file:
    indices = torch.arange(0,10)
    batch = data_utils.Subset(ts_data, indices)
    batch = data_collator(batch)
    batch["inputs_embeds"] = batch["inputs_embeds"].to(device)
    batch["attention_mask"] = batch["attention_mask"].to(device)
    outputs = model.generate(inputs_embeds=batch["inputs_embeds"], attention_mask=batch["attention_mask"], min_length=0, max_length=128)
    for i in range(10):
        file.write(f"ts_data: {i}\n")
        file.write(f"inputs_embeds: {batch['inputs_embeds'][i]}\n")
        label = ts_data[i][1]
        file.write(f"label: {label} \n")
        file.write(f"generated: {tokenizer.decode(outputs[i])} \n")
        file.write(f"generated(skip_special_tokens): {tokenizer.decode(outputs[i], skip_special_tokens=True)} \n")
        file.write("=====\n")

# 10種類文章生成させてみる(closed test)
with open(output_result_dir+"closed_test.txt", "w") as file:
    indices = torch.arange(0,10)
    batch = data_utils.Subset(tr_data, indices)
    batch = data_collator(batch)
    batch["inputs_embeds"] = batch["inputs_embeds"].to(device)
    batch["attention_mask"] = batch["attention_mask"].to(device)
    outputs = model.generate(inputs_embeds=batch["inputs_embeds"], attention_mask=batch["attention_mask"], min_length=0, max_length=128)
    for i in range(10):
        file.write(f"tr_data: {i}\n")
        file.write(f"inputs_embeds: {batch['inputs_embeds'][i]}\n")
        label = tr_data[i][1]
        file.write(f"label: {label} \n")
        file.write(f"generated: {tokenizer.decode(outputs[i])} \n")
        file.write(f"generated(skip_special_tokens): {tokenizer.decode(outputs[i], skip_special_tokens=True)} \n")
        file.write("=====\n")

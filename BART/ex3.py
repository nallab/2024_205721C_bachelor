import joblib
from transformers import BartTokenizer, BartForConditionalGeneration
import torch
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer
from transformers import DataCollatorForSeq2Seq
import numpy as np
import evaluate
from transformers import EarlyStoppingCallback

data_path = "./all_video_i3d_quantify.joblib"
output_path = "./tmp/"
df = joblib.load(data_path)


teachers = list(df["text"])

# 入力の前処理
x = df["i3d_feature_quantiy"]
input_texts = []
for double_list in x:
    all_elements = [element for sublist in double_list for element in sublist]
    input_texts.append(''.join(all_elements))


# トークン追加
new_tokens = []
with open("./tokens.txt", "r") as f:
    for item in f.readlines():
        new_tokens.append(item.rstrip())
print(f"{new_tokens=}")

model_name = "facebook/bart-large"
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
tokenizer.add_tokens(new_tokens)

model.resize_token_embeddings(len(tokenizer))

# 新しいembedding層を取得
new_embedding_layer = model.get_input_embeddings()
# padding_idxを再設定 (通常は0または特定のインデックス)
new_embedding_layer.padding_idx = 1


# データのトークナイズ
inputs = tokenizer(input_texts, padding=True, truncation=True, return_tensors="pt")
with tokenizer.as_target_tokenizer():
    labels = tokenizer(teachers, padding=True, truncation=True, return_tensors="pt")


class Dataset(torch.utils.data.Dataset):
    def __init__(self, features):
        self.val = []
        for i in range(len(features[0]["input_ids"])):
            data_dict = {
                            "input_ids" : features[0]["input_ids"][i] ,
                            "attention_mask" : features[0]["attention_mask"][i],
                            "labels" : features[1]["input_ids"][i],
                        }
            self.val.append(data_dict)

    def __len__(self):
        return len(self.val)

    def __getitem__(self, idx):
        return self.val[idx]

dataset = Dataset([inputs, labels])
# データセットを分割
generator = torch.Generator().manual_seed(42)  # シード固定(同じ結果になるようにする)
tr_data, ev_data, ts_data = torch.utils.data.random_split(dataset, [0.7, 0.2, 0.1], generator=generator)

# datacollatorを用意
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


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


device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# 学習条件
training_args = Seq2SeqTrainingArguments(
    output_dir = output_path, 
    predict_with_generate = True,
    per_device_train_batch_size = 16,
    per_device_eval_batch_size = 16,
    save_strategy = "epoch",
    save_total_limit = 2,
    evaluation_strategy = "epoch",
    logging_strategy = "epoch",
    learning_rate = 5e-5,  # default: 5e-5
    weight_decay = 0.0, 
    warmup_steps = 0,  # default: 0
    do_train = True,
    do_eval = True,
    num_train_epochs=10,
)

trainer = Seq2SeqTrainer(
    model= model,
    args=training_args,
    compute_metrics=compute_metrics_bleu,
    train_dataset=tr_data,
    eval_dataset=ev_data,
    # callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],  ## early_stopping
)

trainer.train()
## →lossなどはtrainer_state.jsonへ格納される


# 10種類文章生成させてみる(open test)
with open(output_path+"open_test.txt", "w") as file:
    for i in range(10):
        input_data = torch.reshape(ts_data[i]["input_ids"], (1, -1)).to(device)
        atm = torch.reshape(ts_data[i]["attention_mask"], (1, -1)).to(device)
        outputs = model.generate(input_ids=input_data, attention_mask=atm, min_length=0, max_length=128)
        file.write(f"ts_data: {i}\n")
        file.write(f"input: {tokenizer.decode(ts_data[i]['input_ids'], skip_special_tokens=True)}\n")
        file.write(f"label: {tokenizer.decode(ts_data[i]['labels'], skip_special_tokens=True)} \n")
        file.write(f"generated: {tokenizer.batch_decode(outputs)} \n")
        file.write(f"generated(skip_special_tokens): {tokenizer.batch_decode(outputs, skip_special_tokens=True)} \n")
        file.write("=====\n")


# 10種類文章生成させてみる(closed test)
with open(output_path+"closed_test.txt", "w") as file:
    for i in range(10):
        input_data = torch.reshape(tr_data[i]["input_ids"], (1, -1)).to(device)
        atm = torch.reshape(tr_data[i]["attention_mask"], (1, -1)).to(device)
        outputs = model.generate(input_ids=input_data, attention_mask=atm, min_length=0, max_length=128)
        file.write(f"tr_data: {i}\n")
        file.write(f"input: {tokenizer.decode(tr_data[i]['input_ids'], skip_special_tokens=True)}\n")
        file.write(f"label: {tokenizer.decode(tr_data[i]['labels'], skip_special_tokens=True)} \n")
        file.write(f"generated: {tokenizer.batch_decode(outputs)} \n")
        file.write(f"generated(skip_special_tokens): {tokenizer.batch_decode(outputs, skip_special_tokens=True)} \n")
        file.write("=====\n")


# testデータに対するBLEU算出
training_args = Seq2SeqTrainingArguments(
    output_dir = output_path, 
    predict_with_generate = True,
    per_device_eval_batch_size = 16,
    do_eval = True,
)
trainer = Seq2SeqTrainer(
    model= model,
    args=training_args,
    compute_metrics=compute_metrics_bleu,
    eval_dataset=ts_data,
)
ts_result = trainer.evaluate()
print(ts_result)
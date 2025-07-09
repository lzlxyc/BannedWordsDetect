import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["ACCELERATE_USE_DEEPSPEED"] = "false"
os.environ["USE_ACCELERATE"] = "false"

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, AutoConfig, ModernBertConfig
)
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy as np


from transformers import TrainerCallback

class SaveBestModelCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % args.save_steps == 0 and state.best_model_checkpoint:
            output_best_dir = os.path.join(args.output_dir, "best_model")
            os.makedirs(output_best_dir, exist_ok=True)
            kwargs["model"].save_pretrained(output_best_dir)


import pandas as pd


def split_long_texts(df, text_col='text', label_col='label', max_len=512, min_len=100):
    new_rows = []
    labels = []
    for _, row in df.iterrows():
        text = row[text_col]
        label = row[label_col]

        # 如果文本长度不超过max_len，直接保留
        if len(text) <= max_len:
            new_rows.append(text)
            labels.append(label)
            continue

        # 切分长文本
        start = 0
        while start < len(text):
            end = start + max_len
            fragment = text[start:end]

            # 只有当片段长度足够时才保留
            if len(fragment) >= min_len:
                new_rows.append(fragment)
                labels.append(label)

            start = end

    # 创建新的DataFrame
    return pd.DataFrame({'text': new_rows, 'label': labels})


def load_df():
    # from utils import class2idx_map
    # df = pd.read_csv('../DATA/train_all.csv')
    # df['text'] = df['文本']
    # df['label'] = df['类别'].map(class2idx_map)
    # df = df.drop(columns=['文本', '类别'])
    # print(df.head())
    # train_data, val_data = train_test_split(
    #     df, test_size=0.05, shuffle=True, random_state=42, stratify=df['label']
    # )
    # train_data.to_csv('../DATA/train.csv', index=False)
    # val_data.to_csv('../DATA/val.csv', index=False)

    def preprocess(text:str) -> str:
        text = text.replace('4','四').replace('6','六').replace('9','九').replace('8','八')
        if len(text) <= 512: return text
        if len(text) < 512*1.5: return text[100:]
        return text[150:]


    train_data = pd.read_csv("../DATA/train.csv")
    val_data = pd.read_csv("../DATA/val.csv")

    train_data['text'] = train_data['text'].apply(preprocess)
    val_data['text'] = val_data['text'].apply(preprocess)

    # print("数据增强前：",len(train_data))
    # print(train_data.head())
    # train_data = split_long_texts(train_data)
    # print("数据增强后：",len(train_data))
    # print(train_data.head())

    return train_data, val_data



def load_data(tokenize_path):
    def tokenize_func(examples):
        return tokenizer(examples["text"], truncation=True, max_length=MAX_LENGTH, padding="max_length")

    tokenizer = AutoTokenizer.from_pretrained(tokenize_path)  # 选中文预训练模型

    train_data, val_data = load_df()
    datasets_train, datasets_val = Dataset.from_pandas(train_data), Dataset.from_pandas(val_data)

    train_tokenized_ds = datasets_train.map(tokenize_func, batched=True)  # 批量分词
    val_tokenized_ds = datasets_val.map(tokenize_func, batched=True)  # 批量分词

    return train_tokenized_ds, val_tokenized_ds


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"f1_macro": round(f1_score(labels, predictions, average="macro"), 4)}


def dotrain(model_path, tokenize_path):
    # 1. 数据准备（假设 df 有 'text' 文本列、'label' 类别列 ）
    train_tokenized_ds, val_tokenized_ds = load_data(tokenize_path)


    config = ModernBertConfig.from_pretrained(model_path)  # 假设有两个分类类别
    config.max_position_embeddings = MAX_LENGTH
    config.num_labels=10

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        config=config,
        ignore_mismatched_sizes=True  # 忽略权重不匹配警告
    )

    # model = AutoModelForSequenceClassification.from_pretrained(
    #     model_path,
    #     num_labels=10,
    #     ignore_mismatched_sizes=True  # 忽略权重不匹配警告
    # )


    # 3. 训练参数配置（小学习率 + 权重衰减防过拟合 ）
    training_args = TrainingArguments(
        output_dir=f'{SAVE_MODEL_DIR}/{use_model}_v2',
        learning_rate=0.0001, # 2e-5
        per_device_train_batch_size=8, # 32
        per_device_eval_batch_size=24, # 64
        do_train=True,
        do_eval=True,
        num_train_epochs=5,
        eval_strategy="steps",  # 修改为新的参数名
        save_strategy="steps",
        logging_steps=100,
        eval_steps=200,
        save_steps=200,
        save_total_limit=3,
        # weight_decay=0.01,  # 正则化
        report_to="none",
        load_best_model_at_end=True,  # Optional: to load the best model at the end
        metric_for_best_model="f1_macro",  # Use macro F1 to determine the best model
        greater_is_better=True,  # Since higher F1 is better
        # disable_tqdm=True  # 添加这行来禁用所有进度条
    )

    # 4. 训练与评估
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized_ds,
        eval_dataset=val_tokenized_ds,
        compute_metrics=compute_metrics,
        callbacks=[SaveBestModelCallback()],  # 添加回调
    )
    trainer.train()  # 开始微调

    trainer.save_model(f"{SAVE_MODEL_DIR}/{use_model}_v2/best_model")
    print(f"Model saved to:{SAVE_MODEL_DIR}/{use_model}_v2/best_model")

if __name__ == "__main__":
    MODEL_ROOT_DIR = 'D:/LZL/workspace/ModelHub'
    SAVE_MODEL_DIR = 'D:/LZL/workspace/XunfeiCom2025/models'
    MAX_LENGTH = 512

    use_model = 'chinese_roberta_L-10_H-768'
    tokenize_path = f'{MODEL_ROOT_DIR}/{use_model}'
    # model_path = 'D:/LZL/workspace/XunfeiCom2025/models/chinese_roberta_L-10_H-768/checkpoint-4100'
    model_path = tokenize_path

    dotrain(model_path, tokenize_path)

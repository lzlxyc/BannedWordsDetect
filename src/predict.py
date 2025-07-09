from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

def batch_predict(texts, model_path, batch_size=32, device="cuda"):
    """
    分批次预测大批量文本

    Args:
        texts: 待预测文本列表
        model_path: 模型路径
        batch_size: 每批处理数量
        device: 指定设备(cpu/cuda)

    Returns:
        all_predictions: 所有预测结果
        all_probabilities: 所有概率分布
    """
    # 加载模型和tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenize_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()

    all_predictions = []
    all_probabilities = []

    # 分批处理
    for i in tqdm(range(0, len(texts), batch_size), desc="Predicting"):
        batch_texts = texts[i:i + batch_size]

        # Tokenize
        inputs = tokenizer(
            batch_texts,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt"
        ).to(device)

        # 预测
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

        # 收集结果
        if device == "cpu":
            all_predictions.extend(preds.tolist())
            all_probabilities.extend(probs.numpy())
        else:
            all_predictions.extend(preds.cpu().tolist())
            all_probabilities.extend(probs.cpu().numpy())

    return all_predictions, all_probabilities


def do_eval():
    from utils import idx2class_map
    df = pd.read_csv("../DATA/val.csv").head(200)
    all_predictions, all_probabilities = batch_predict(df['text'].tolist(), model_path)
    # print(all_predictions)
    # print(df['label'].tolist())
    # for i,j in zip(all_predictions, df['label'].tolist()):
    #     if i != j:
    #         print(i, j)

    df['pred'] = [idx2class_map[i] for i in all_predictions]
    df['label'] = df['label'].map(idx2class_map)

    df.to_csv("../DATA/pred.csv", index=False)

    f1 = round(f1_score(df['label'].tolist(), df['pred'].tolist(), average="macro"), 4)

    report = classification_report(df['label'].tolist(), df['pred'].tolist(), output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df = report_df.reset_index().rename(columns={'index': 'class'})
    report_df.to_csv("../DATA/report.csv", index=False)

    # _info = pd.DataFrame(_info)[['f1-score']]
    print(f'***** f1 score::{f1}')

    report_df['f1-score'] = report_df['f1-score'].apply(lambda x: round(x, 4))
    report_df = dict(report_df[['class','f1-score']].values)
    for k, v in report_df.items():
        print(f'{k}: {v}')





def do_test():
    from utils import idx2class_map
    df = pd.read_csv("../DATA/test_text.csv")
    all_predictions, all_probabilities = batch_predict(df['文本'].tolist(), model_path)
    # print(all_predictions)
    # print(df['label'].tolist())
    # for i,j in zip(all_predictions, df['label'].tolist()):
    #     if i != j:
    #         print(i, j)

    df['类别'] = [idx2class_map[i] for i in all_predictions]

    df[['id','类别']].to_csv("../DATA/submit.csv", index=False)
    df.to_csv("../DATA/all_submit.csv", index=False)

    print(f'完成预测...')


if __name__ == "__main__":
    tokenize_path = "D:/LZL/workspace/ModelHub/chinese_roberta_L-10_H-768"
    model_path = 'D:/LZL/workspace/XunfeiCom2025/models/chinese_roberta_L-10_H-768_v2/best_model'
    do_eval()

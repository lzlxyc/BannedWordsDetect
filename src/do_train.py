from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
import torch
from torch.optim import AdamW
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import f1_score

from models import TextClassifier
from data_sets import load_data


# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained("D:/LZL/workspace/ModelHub/chinese_roberta_L-10_H-768")
model = AutoModel.from_pretrained("D:/LZL/workspace/ModelHub/chinese_roberta_L-10_H-768")
# 初始化分类器
classifier = TextClassifier(model)

train_dataloader, val_dataloader = load_data(tokenizer)

# 训练配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classifier.to(device)

optimizer = AdamW(classifier.parameters(), lr=2e-5)
epochs = 10
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)


# 训练循环
def train_epoch(model, dataloader, optimizer, scheduler, device, val_dataloader, epoch, best_f1):
    model.train()
    total_loss = 0

    step = 1
    eval_steps = min(100, len(dataloader) // 7)
    log_steps = eval_steps // 2

    for batch in tqdm(dataloader, total=len(dataloader)):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        outputs = model({
            'input_ids': input_ids,
            'attention_mask': attention_mask
        })

        loss = nn.CrossEntropyLoss()(outputs, labels)
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % eval_steps == 0:
            F1 = evaluate(model, val_dataloader, device)
            if F1 > best_f1:
                print(f'***********************************Best F1 on val set: {best_f1} ===>> {F1}')
                best_f1 = F1
                torch.save(classifier.state_dict(), '../models/best_text_classifier.pth')
        elif step % log_steps == 0:
            print(f'Epoch {epoch} train loss {total_loss/step}')

        step += 1

    # avg_loss = total_loss / len(dataloader)
    return best_f1


# 验证循环
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model({
                'input_ids': input_ids,
                'attention_mask': attention_mask
            })

            loss = nn.CrossEntropyLoss()(outputs, labels)
            total_loss += loss.item()

            _, predictions = torch.max(outputs, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)

            all_labels += labels.tolist()
            all_predictions += predictions.tolist()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_predictions
    F1 = f1_score(all_labels, all_predictions, average='macro')

    print(f"Accuracy: {accuracy} F1: {F1} eval loss: {avg_loss}")
    return F1


def trainer():
    # 训练模型
    print("开始训练模型...")
    best_f1 = 0.0

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        F1 = train_epoch(classifier, train_dataloader, optimizer, scheduler, device, val_dataloader, epoch, best_f1)
        if F1 > best_f1:
            best_f1 = F1

    # 保存模型
    torch.save(classifier.state_dict(), '../models/text_classifier.pth')
    print("模型训练完成并保存!")



if __name__ == "__main__":
    trainer()
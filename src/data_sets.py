import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# 模拟数据集
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        # print(torch.tensor(label, dtype=torch.long))
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }



def load_data(tokenizer):
    # 模拟数据
    # texts = [
    #     "这是一个积极的句子，充满了正能量。",
    #     "这是一个消极的句子，感觉非常糟糕。",
    #     "今天天气真好，阳光明媚。",
    #     "这个电影太无聊了，浪费时间。",
    #     "我喜欢这个产品，质量非常好。",
    #     "这个服务太差劲了，非常不满意。",
    #     "大模型对程序员来说是一个很好的工具。",
    #     "大模型对初级开发者来说是一个坏消息。"
    # ]
    #
    # #  PyTorch 的交叉熵损失函数 nn.CrossEntropyLoss 要求标签必须是从 0 开始的连续整数（如 0、1、2...）
    # labels = [1, 0, 1, 0, 1, 0, 1, 0]  # 1表示积极，0表示消极

    from utils import class2idx_map
    df = pd.read_csv('../DATA/train_all.csv')
    texts = df['文本'].tolist()
    labels = df['类别'].map(class2idx_map).tolist()

    # 划分训练集和验证集
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.05, random_state=42)

    print(len(train_texts), len(val_texts))
    print(train_texts[:5], train_labels[:5])

    # 创建数据加载器
    train_dataset = TextDataset(train_texts, train_labels, tokenizer)
    val_dataset = TextDataset(val_texts, val_labels, tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32)

    return train_dataloader, val_dataloader


if __name__ == '__main__':
    load_data(None)

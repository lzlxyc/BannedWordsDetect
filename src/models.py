import torch.nn as nn


# 定义分类模型
class TextClassifier(nn.Module):
    def __init__(self, model, hidden_size=768, num_classes=10):
        super().__init__()
        self.model = model
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, inputs):
        # 获取模型输出
        outputs = self.model(**inputs)
        # 使用 [CLS] token
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        # 通过分类器
        logits = self.classifier(cls_embedding)
        return logits

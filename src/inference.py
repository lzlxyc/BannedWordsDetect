from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
import numpy as np

from models import TextClassifier

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
base_model = AutoModel.from_pretrained("BAAI/bge-m3")

# 初始化分类器
classifier = TextClassifier(base_model)
classifier.to(device)

# 加载保存的模型权重
try:
    classifier.load_state_dict(torch.load('text_classifier.pth', map_location=device))
    print("模型加载成功!")
except FileNotFoundError:
    print("错误: 找不到模型文件 'text_classifier.pth'。请确保该文件在正确的路径下。")
    exit()

# 设置为评估模式
classifier.eval()


# 预测函数
def predict_sentiment(text):
    """预测文本的情感极性（积极或消极）"""
    # 预处理文本
    inputs = tokenizer(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    # 将输入移至设备
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 进行预测
    with torch.no_grad():
        outputs = classifier(inputs)
        # 获取预测概率
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        # 获取预测类别
        predicted_class = torch.argmax(probabilities, dim=1).item()

    # 类别映射
    sentiment_map = {0: "消极", 1: "积极"}

    return {
        "text": text,
        "predicted_class": predicted_class,
        "sentiment": sentiment_map[predicted_class],
        "confidence": probabilities[0][predicted_class].item()
    }


# 示例预测
if __name__ == "__main__":
    # 测试几个例子
    test_texts = [
        "这个产品真的太棒了，我非常满意！",
        "这个服务太糟糕了，简直是浪费时间。",
        "今天天气真好，适合出去散步。",
        "这个电影很无聊，不推荐观看。"
    ]

    for text in test_texts:
        result = predict_sentiment(text)
        print(f"文本: {result['text']}")
        print(f"情感: {result['sentiment']} ({result['confidence']:.4f})")
        print("-" * 50)

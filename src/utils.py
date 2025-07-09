import torch

idx2class_map = {0: '种族歧视',
                 1: '政治敏感',
                 2: '微侵犯(MA)',
                 3: '色情',
                 4: '犯罪',
                 5: '地域歧视',
                 6: '基于文化背景的刻板印象(SCB)',
                 7: '宗教迷信',
                 8: '性侵犯(SO)',
                 9: '基于外表的刻板印象(SA)'}

class2idx_map = {'种族歧视': 0,
                 '政治敏感': 1,
                 '微侵犯(MA)': 2,
                 '色情': 3,
                 '犯罪': 4,
                 '地域歧视': 5,
                 '基于文化背景的刻板印象(SCB)': 6,
                 '宗教迷信': 7,
                 '性侵犯(SO)': 8,
                 '基于外表的刻板印象(SA)': 9}


def convert(tokenizer, max_length, text, label):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    print(torch.tensor(label, dtype=torch.long))

    return {
        'input_ids': encoding['input_ids'].flatten(),
        'attention_mask': encoding['attention_mask'].flatten(),
        'label': torch.tensor(label, dtype=torch.long)
    }
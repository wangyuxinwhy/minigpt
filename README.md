# MiniGPT

Better MinGPT
1. 更规范的变量命名
2. 更详尽的类型注解
3. 更简单的代码实现

# Usage

1. 使用 GPTModel
```
import torch
from minigpt.model import GPTForLanguageModel

model = GPTForLanguageModel.from_pretrained('gpt2')
input_ids = torch.tensor([[1, 2, 3, 4, 5]])
logits = model(input_ids)['logits']
```
2. 生成文本
```
from minigpt.text_generator import GPTTextGernerator

text_generator = GPTTextGernerator.from_pretrained('gpt2')
print(text_generator.generate('My name is Clara and I am'))
```

# TODO
- [ ] 实现自己的 BPETokenizer,去除掉 transformers 的依赖

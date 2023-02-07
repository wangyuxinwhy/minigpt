from __future__ import annotations

from typing import cast

import torch
from transformers import GPT2Tokenizer

from minigpt.model import GPTForLanguageModel


class GPTTextGernerator:

    def __init__(
        self,
        model: GPTForLanguageModel,
        tokenizer: GPT2Tokenizer,
    ):
        self.model = model
        self.tokenizer = tokenizer
    
    def generate(self, prompt_text: str, num_new_words: int=100, temperature: float=1.0) -> str:
        input_ids = self.tokenizer(prompt_text, return_tensors='pt')['input_ids']
        input_ids = cast(torch.LongTensor, input_ids)
        output_ids = self.model.generate(input_ids, num_new_tokens=num_new_words, temperature=temperature)
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, tokenizer_path: str | None=None) -> GPTTextGernerator:
        tokenizer_path = tokenizer_path or model_name_or_path
        tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
        model = GPTForLanguageModel.from_pretrained(model_name_or_path)
        return cls(model, tokenizer)

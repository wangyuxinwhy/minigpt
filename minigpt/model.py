from __future__ import annotations

import math

import torch
import torch.nn as nn


class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        max_length: int,
        dropout_prob: float=0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.max_length = max_length
        self.dropout_prob = dropout_prob

        self.input_projection = nn.Linear(in_features=hidden_size, out_features=3*hidden_size)
        self.output_projection = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.attention_dropout = nn.Dropout(p=dropout_prob)
        self.output_dropout = nn.Dropout(p=dropout_prob)
        self.causal_mask = torch.tril(torch.ones(max_length, max_length)).view(1, 1, max_length, max_length)

    def forward(self, input_tensor: torch.Tensor):
        # input_tensor: [batch_size, sequence_length, hidden_size]
        batch_size, sequence_length, hidden_size = input_tensor.size()

        # projected tensor: [batch_size, sequence_length, 3*hidden_size]
        projected_tensor = self.input_projection(input_tensor)

        # query, key, value: [batch_size, sequence_length, hidden_size]
        query, key, value = torch.chunk(projected_tensor, chunks=3, dim=-1)

        hidden_size_per_head = hidden_size // self.num_heads
        # query: [batch_size, num_heads, sequence_length, hidden_size_per_head]
        query = query.view(batch_size, sequence_length, self.num_heads, hidden_size_per_head).transpose(1, 2)
        # key: [batch_size, num_heads, hidden_size_per_head, sequence_length]
        key = key.view(batch_size, sequence_length, self.num_heads, hidden_size_per_head).transpose(1, 2).transpose(-2, -1)
        # value: [batch_size, num_heads, sequence_length, hidden_size_per_head]
        value = value.view(batch_size, sequence_length, self.num_heads, hidden_size_per_head).transpose(1, 2)

        # attention_scores: [batch_size, num_heads, sequence_length, sequence_length]
        attention_scores = torch.matmul(input=query, other=key) / math.sqrt(hidden_size_per_head)
        masked_attention_scores = attention_scores.masked_fill(self.causal_mask[:,:,:sequence_length,:sequence_length] == 0, float('-inf'))
        attention_prob = torch.nn.functional.softmax(masked_attention_scores, dim=-1)
        attention_prob = self.attention_dropout(attention_prob)

        # output: [batch_size, num_headers, sequence_length, hidden_size_per_head]
        output = torch.matmul(attention_prob, value)
        # output: [batch_size, sequence_length, hidden_size]
        output = output.transpose(1, 2).contiguous().view(batch_size, sequence_length, hidden_size)
        output = self.output_dropout(self.output_projection(output))
        return output


class FeedForward(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        dropout_prob: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(in_features=hidden_size, out_features=4*hidden_size)
        self.activation = NewGELU()
        self.output_projection = nn.Linear(in_features=4*hidden_size, out_features=hidden_size)
        self.dropout = nn.Dropout(p=dropout_prob)
    
    def forward(self, input_tensor: torch.Tensor):
        # input_tensor: [batch_size, sequence_length, hidden_size]
        output = self.output_projection(self.activation(self.input_projection(input_tensor)))
        output = self.dropout(output)
        return output


class GPTBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        max_length: int,
        dropout_prob: float=0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.max_length = max_length
        self.dropout_prob = dropout_prob

        self.input_layer_norm = nn.LayerNorm(hidden_size)
        self.intermediate_layer_norm = nn.LayerNorm(hidden_size)

        self.attention = CausalSelfAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            max_length=max_length,
            dropout_prob=dropout_prob,
        )

        self.feedforward = FeedForward(hidden_size=hidden_size, dropout_prob=dropout_prob)

    def forward(self, input_tensor: torch.Tensor):
        # input_tensor: [batch_size, sequence_length, hidden_size]
        # output: [batch_size, sequence_length, hidden_size]
        output = input_tensor + self.attention(self.input_layer_norm(input_tensor))
        output = output + self.feedforward(self.intermediate_layer_norm(output))
        return output


class GPTModel(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_heads: int,
        max_length: int,
        num_layers: int,
        dropout_prob: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.num_layers = num_layers
        self.dropout_prob=dropout_prob

        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=hidden_size
        )
        self.position_embedding = nn.Embedding(
            num_embeddings=max_length, embedding_dim=hidden_size
        )
        self.dropout = nn.Dropout(p=dropout_prob)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.blocks = nn.ModuleList([
            GPTBlock(hidden_size=hidden_size, num_heads=num_heads, max_length=max_length, dropout_prob=dropout_prob)
            for _ in range(num_layers)
        ])


    def forward(self, input_ids: torch.Tensor):
        device = input_ids.device
        sequence_length = input_ids.size(1)

        positions_ids = torch.arange(0, sequence_length, dtype=torch.long, device=device).unsqueeze(0)
        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(positions_ids)

        hidden_states = self.dropout(token_embeddings + position_embeddings)
        for block in self.blocks:
            hidden_states = block(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


class GPTForLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_heads: int,
        max_length: int,
        num_layers: int,
        dropout_prob: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.num_layers = num_layers
        self.dropout_prob=dropout_prob

        self.transformer = GPTModel(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            max_length=max_length,
            num_layers=num_layers,
            dropout_prob=dropout_prob
        )
        self.language_model_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.criterion = nn.CrossEntropyLoss()
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, parameter in module.named_parameters():
            if name == "output_projection.weight":
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                torch.nn.init.normal_(
                    parameter,
                    mean=0.0,
                    std=(0.02 / math.sqrt(2 * self.num_layers))
                )
    
    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None):
        # input_ids, labels: [batch_size, sequence_length]
        # hidden_states: [batch_size, sequence_length, hidden_size]
        hidden_states = self.transformer(input_ids)
        # logits: [batch_size, sequence_length, vocab_size]
        logits = self.language_model_head(hidden_states)
        output = {
            'logits': logits,
        }
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = self.criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            output['loss'] = loss
        return output

    @classmethod
    def from_pretrained(cls, model_name_or_path: str) -> GPTForLanguageModel:
        from typing import cast
        from transformers import GPT2LMHeadModel
        from minigpt.utils import adapt_huggingface_transformers_state_dict

        huggingface_model = GPT2LMHeadModel.from_pretrained(model_name_or_path)
        huggingface_model = cast(GPT2LMHeadModel, huggingface_model)
        model = GPTForLanguageModel(
            vocab_size=huggingface_model.config.vocab_size,
            hidden_size=huggingface_model.config.n_embd,
            num_heads=huggingface_model.config.n_head,
            max_length=huggingface_model.config.n_ctx,
            num_layers=huggingface_model.config.n_layer,
            dropout_prob=huggingface_model.config.attn_pdrop,
        )
        state_dict = huggingface_model.state_dict()
        state_dict = adapt_huggingface_transformers_state_dict(state_dict)
        model.load_state_dict(state_dict)
        model = model.eval()
        return model

    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor, num_new_tokens: int=64, temperature: float=1.0):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(num_new_tokens):
            logits = self(prompt_ids)['logits']
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            new_token_id = torch.multinomial(probs, num_samples=1)
            prompt_ids = torch.cat((prompt_ids, new_token_id), dim=1)
        return prompt_ids

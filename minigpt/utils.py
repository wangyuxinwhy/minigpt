def _adapt_key(key: str) -> str:
    key_map = {
        'wte': 'token_embedding',
        'wpe': 'position_embedding',
        'h': 'blocks',
        'attn': 'attention',
        'ln_f': 'layer_norm',
        'ln_1': 'input_layer_norm',
        'ln_2': 'intermediate_layer_norm',
        'c_attn': 'input_projection',
        'c_fc': 'input_projection',
        'c_proj': 'output_projection',
        'mlp': 'feedforward',
        'lm_head': 'language_model_head',
    }
    items = key.split('.')
    new_items = []
    for item in items:
        if item in key_map:
            new_items.append(key_map[item])
        else:
            new_items.append(item)
    return '.'.join(new_items)


def adapt_huggingface_transformers_state_dict(state_dict: dict):
    new_state_dict = {}
    transpose_names = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
    remove_names = ['.attn.bias', '.attn.masked_bias']
    for key, value in state_dict.items():
        if any(key.endswith(name) for name in remove_names):
            continue

        new_key = _adapt_key(key)
        if any(key.endswith(name) for name in transpose_names):
            new_state_dict[new_key] = value.t()
        else:
            new_state_dict[new_key] = value
    return new_state_dict

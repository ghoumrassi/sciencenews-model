import re


def unfreeze_n_layers(model, n: int):
    # Freeze all layers in the model
    for _, param in model.named_parameters():
        param.requires_grad = False

    pattern = r'transformer\.layer\.(\d+)\.'

    last_layer = max(int(re.search(pattern, name).group(1))
                     for name, _ in model.named_parameters()
                     if re.search(pattern, name))

    # Unfreeze the last n layers in the model
    for name, param in model.named_parameters():
        match = re.search(pattern, name)
        if not match:
            continue

        layer_index = int(match.group(1))
        if layer_index > last_layer - n:
            param.requires_grad = True

    return model

def count_trainable_params(model):
    if hasattr(model, 'get_nb_trainable_parameters'):
        trainable, _ = model.get_nb_trainable_parameters()
    else:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
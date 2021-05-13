import torch.optim as Optim
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup as WarmupLinearSchedule

def get_Adam_optim(__C, model):
    return AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=__C.TRAIN.lr_base,
    )

def get_Adam_optim_v1(__C, model):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': filter(lambda x: x.requires_grap, [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)]), 'weight_decay': 0.01},
        {'params': filter(lambda x: x.requires_grap, [p for n, p in param_optimizer if any(nd in n for nd in no_decay)]), 'weight_decay': 0.0}]
    return Optim.AdamW(
        optimizer_grouped_parameters,
        lr=__C.TRAIN.lr_base,
        weight_decay=0.01,
    )

def get_Adam_optim_v2(config, model):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.TRAIN.lr_base, weight_decay=0.01, correct_bias=False)
    scheduler = WarmupLinearSchedule(optimizer, num_training_steps=config.TRAIN.num_train_optimization_steps,
                                     num_warmup_steps=config.TRAIN.warmup_proportion * config.TRAIN.num_train_optimization_steps)
    return optimizer, scheduler
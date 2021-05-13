import os
import torch
import random
import numpy as np
import os.path as osp
from easydict import EasyDict as edict

__C = edict()

cfg = __C

# ======================================= #
# ------------ Common options ----------- #
# ======================================= #
__C.max_length = 256
__C.attr_dim = 768
__C.num_attr_heads = 12
__C.num_attrs = 3
__C.layer_norm_eps = 1e-12
__C.hidden_dropout_prob = 0.2
__C.attention_probs_dropout_prob = 0.2
__C.hidden_size = 768
__C.intermediate_size = __C.hidden_size * __C.num_attrs
__C.hidden_act = 'gelu'
__C.max_position_embeddings = 30
__C.n_mmalayer = 6
__C.n_delayer = 12
__C.n_bertlayer = 12
__C.n_totallayer =12
__C.type = 'd' # a, b, c, d

# ======================================= #
# ----------- Training options ---------- #
# ======================================= #
__C.TRAIN = edict()

__C.TRAIN.batch_size = 4

__C.TRAIN.lr_base = 2e-5

__C.TRAIN.weight_decay = 0.0001

__C.TRAIN.early_stop = True

__C.TRAIN.opt_eps = 1e-9

__C.TRAIN.opt_betas = (0.9, 0.98)

__C.TRAIN.momentum = 0.9

__C.TRAIN.resume_snapshot = ''

__C.TRAIN.max_epoch = 20

__C.TRAIN.patience = 3

__C.TRAIN.max_grad_norm = 2.0

__C.TRAIN.warmup_proportion = 0.1

__C.TRAIN.gradient_accumulation_steps = 4

__C.TRAIN.num_train_optimization_steps = None

# ======================================= #
# ------------- Model options ----------- #
# ======================================= #
__C.ckpts_path = 'ckpts'

__C.seed = random.randint(0, 99999999)

__C.version = 'default'

__C.model = 'model'

__C.model_save_path = osp.join(__C.ckpts_path, __C.version)

__C.log_path = 'logs'

# ======================================= #
# ------------- TEST options ------------ #
# ======================================= #
__C.TEST = edict()

__C.TEST.batch_size = 4

# ======================================= #
# ------------- Basic options ----------- #
# ======================================= #
__C.BASE = edict()

__C.BASE.max_length = 512

__C.BASE.strategy = 'both'


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print(('Error under config key: {}'.format(k)))
                raise
        else:
            b[k] = v


def add_edit(a, b):
    if type(a) is not edict:
        return

    for k, v in a.items():
        b[k] = v


def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert subkey in d
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(
                type(value), type(d[subkey]))
        d[subkey] = value


def proc(config):
    assert config.run_mode in ['train', 'val', 'test', '']

    # ------------- Devices setup ----------- #
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu
    gpus = config.gpu.split(',')
    config.n_gpu = len(gpus)
    config.device = "cuda" if config.n_gpu else "cpu"

    # -------------  Seed setup  ----------- #
    # fix pytorch seed
    torch.manual_seed(config.seed)
    if config.n_gpu < 2:
        torch.cuda.manual_seed(config.seed)
    else:
        torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = True


    # fix numpy seed
    np.random.seed(config.seed)

    # fix random seed
    random.seed(config.seed)

    # initial directors if not exist
    if not os.path.exists(config.log_path):
        os.makedirs(config.log_path)
    if not os.path.exists(config.ckpts_path):
        os.makedirs(config.ckpts_path)


def config_print(config):
    for k, v in config.items():
        print('{ %-17s }->' % k, v)
    return ''

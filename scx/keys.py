import torch
import numpy as np


class SCXKeyGenerator:
    def __init__(self, seq_len, hidden_dim, qk_hidden_dim, redundant_num=0, alp=False, 
                 batch_size=1, type=torch.float16, 
                 decode_start_layers=None, decode_end_layers=None):
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.qk_hidden_dim = qk_hidden_dim
        self.redundant_num = redundant_num
        self.alp = alp
        self.batch_size = batch_size
        self.type = type

        self.decode_start_layers = decode_start_layers if decode_start_layers is not None else []
        self.decode_end_layers = decode_end_layers if decode_end_layers is not None else []

    def gen_keys(self, layer_idx):

        r_embeds = None
        if self.redundant_num > 0:
            r_embeds = np.random.randn(self.batch_size, self.redundant_num, self.hidden_dim)
            r_embeds = torch.from_numpy(r_embeds).to(dtype=self.type)

        attn_pi_left = np.random.permutation(self.seq_len + self.redundant_num)
        attn_inv_pi_left = inv_permutation(attn_pi_left)
        attn_pi_right = np.random.permutation(self.qk_hidden_dim)
        attn_inv_pi_right = inv_permutation(attn_pi_right)

        wo_pi_left = np.random.permutation(self.seq_len)
        wo_inv_pi_left = inv_permutation(wo_pi_left)

        ff_pi_left = np.random.permutation(self.seq_len)
        ff_inv_pi_left = inv_permutation(ff_pi_left)

        dec_attn_alp = None
        dec_attn_pi_right = None
        dec_attn_inv_pi_right = None
        
        if layer_idx in self.decode_start_layers:
            # dec_attn_alp = np.random.randn(self.batch_size, 1, self.hidden_dim)
            dec_attn_alp = np.zeros((self.batch_size, 1, self.hidden_dim)) # just for testing
            dec_attn_alp = torch.from_numpy(dec_attn_alp).to(dtype=self.type)

            dec_attn_pi_right = np.random.permutation(self.qk_hidden_dim)
            dec_attn_inv_pi_right = inv_permutation(dec_attn_pi_right)
        
        if layer_idx in self.decode_end_layers:
            dec_attn_alp = np.random.randn(self.batch_size, 1, self.hidden_dim)

        keys = {
            # prefill mode
            "attn_pi_left": attn_pi_left,
            "attn_inv_pi_left": attn_inv_pi_left,
            "attn_pi_right": attn_pi_right,
            "attn_inv_pi_right": attn_inv_pi_right,
            "wo_pi_left": wo_pi_left,
            "wo_inv_pi_left": wo_inv_pi_left,
            "r_embeds": r_embeds,
            "ff_pi_left": ff_pi_left,
            "ff_inv_pi_left": ff_inv_pi_left,
            # decode mode
            "dec_attn_alp": dec_attn_alp,
            "dec_attn_pi_right": dec_attn_pi_right,
            "dec_attn_inv_pi_right": dec_attn_inv_pi_right,
        }
        return keys


def inv_permutation(p):
    inv_p = [0]*len(p)
    for old_idx, new_idx in enumerate(p):
        inv_p[new_idx] = old_idx
    return inv_p


def copy_submodules(src_module, tgt_module, debug=False):
    for name, src_child in src_module.named_children():
        setattr(tgt_module, name, src_child)
        if debug:
            print(f"Copied {name}")
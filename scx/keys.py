import torch
import numpy as np


class SCXKeyGenerator:
    def __init__(self, seq_len, hidden_dim, qk_hidden_dim, redundant_num=0, alp=False, batch_size=1, type=torch.float16):
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.qk_hidden_dim = qk_hidden_dim
        self.redundant_num = redundant_num
        self.alp = alp
        self.batch_size = batch_size
        self.type = type

    def gen_keys(self):
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

        alpha = None
        if self.alp:
            alpha = np.random.randn(self.batch_size, self.seq_len + self.redundant_num, self.hidden_dim)
            alpha = torch.from_numpy(alpha).to(dtype=self.type)

        keys = {
            "attn_pi_left": attn_pi_left,
            "attn_inv_pi_left": attn_inv_pi_left,
            "attn_pi_right": attn_pi_right,
            "attn_inv_pi_right": attn_inv_pi_right,
            "wo_pi_left": wo_pi_left,
            "wo_inv_pi_left": wo_inv_pi_left,
            "r_embeds": r_embeds,
            "alpha": alpha,
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
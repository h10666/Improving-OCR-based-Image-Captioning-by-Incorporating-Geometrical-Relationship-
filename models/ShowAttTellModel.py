# This file contains Att2in2, AdaAtt, AdaAttMO, TopDown model

# AdaAtt is from Knowing When to Look: Adaptive Attention via A Visual Sentinel for Image Captioning
# https://arxiv.org/abs/1612.01887
# AdaAttMO is a modified version with maxout lstm

# Att2in is from Self-critical Sequence Training for Image Captioning
# https://arxiv.org/abs/1612.00563
# In this file we only have Att2in2, which is a slightly different version of att2in,
# in which the img feature embedding and word embedding is the same as what in adaatt.

# TopDown is from Bottom-Up and Top-Down Attention for Image Captioning and VQA
# https://arxiv.org/abs/1707.07998
# However, it may not be identical to the author's architecture.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import misc.utils as utils
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

from .CaptionModel import CaptionModel
from .AttModel import *
from .AttModel import sort_pack_padded_sequence, pad_unsort_packed_sequence, pack_wrapper, _batch_gather

class ShowAttTellCore(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(ShowAttTellCore, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm

        self.lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size * 2, opt.rnn_size)
        self.attention = Attention(opt)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, att_masks=None):
        prev_att = state[0][1]
        lstm_input = torch.cat([prev_att, torch.mean(att_feats, 1), xt], 1)

        h, c = self.lstm(lstm_input, (state[0][0], state[1][0]))

        att = self.attention(h, att_feats, p_att_feats, att_masks)

        output = F.dropout(h, self.drop_prob_lm, self.training)
        state = (torch.stack([h, att]), torch.stack([c, state[1][1]]))

        return output, state

class ShowAttTellModel(AttModel):
    def __init__(self, opt):
        super(ShowAttTellModel, self).__init__(opt)
        self.num_layers = 2
        self.core = ShowAttTellCore(opt)
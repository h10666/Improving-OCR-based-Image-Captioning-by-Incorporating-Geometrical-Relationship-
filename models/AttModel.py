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
from .TransformerModel import Encoder, EncoderLayer, MultiHeadedAttention, PositionwiseFeedForward, LayerNorm

def sort_pack_padded_sequence(input, lengths):
    sorted_lengths, indices = torch.sort(lengths, descending=True)
    tmp = pack_padded_sequence(input[indices], sorted_lengths, batch_first=True)
    inv_ix = indices.clone()
    inv_ix[indices] = torch.arange(0,len(indices)).type_as(inv_ix)
    return tmp, inv_ix

def pad_unsort_packed_sequence(input, inv_ix):
    tmp, _ = pad_packed_sequence(input, batch_first=True)
    tmp = tmp[inv_ix]
    return tmp

def pack_wrapper(module, att_feats, att_masks):
    if att_masks is not None:
        packed, inv_ix = sort_pack_padded_sequence(att_feats, att_masks.data.long().sum(1))
        return pad_unsort_packed_sequence(PackedSequence(module(packed[0]), packed[1]), inv_ix)
    else:
        return module(att_feats)

def _batch_gather(x, inds):
    assert x.dim() == 3
    batch_size = x.size(0)
    length = x.size(1)
    dim = x.size(2)
    x_flat = x.view(batch_size*length, dim)

    batch_offsets = torch.arange(batch_size, device=inds.device) * length
    if inds.dim() == 2:
        batch_offsets = batch_offsets.unsqueeze(-1)
    assert batch_offsets.dim() == inds.dim()
    inds_flat = batch_offsets + inds
    results = F.embedding(inds_flat, x_flat)
    return results

class OcrPtrNet(nn.Module):
    def __init__(self, query_size, key_size=None, query_key_size=None, label_smoothing=0.0):
        super(OcrPtrNet, self).__init__()

        if query_key_size is None:
            query_key_size = query_size
        if key_size is None:
            key_size = query_size
        self.query_key_size = query_key_size
        self.label_smoothing = label_smoothing

        self.query = nn.Linear(query_size, query_key_size)
        self.key = nn.Linear(key_size, query_key_size)

    def forward(self, query_inputs, key_inputs, attention_mask):
        
        query_layer = self.query(query_inputs).unsqueeze(1)
        key_layer = self.key(key_inputs)

        scores = torch.matmul(
            query_layer,
            key_layer.transpose(-1, -2)
        )
        scores = scores.squeeze(1) / math.sqrt(self.query_key_size)

        if self.label_smoothing == 0:
            extended_attention_mask = (1.0 - attention_mask) * -1e12
            scores = scores + extended_attention_mask

        return scores         

class AttModel(CaptionModel):
    def __init__(self, opt):
        super(AttModel, self).__init__()
        self.opt = opt
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        #self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size
        self.ocr_size = opt.ocr_size
        self.label_smoothing = opt.label_smoothing
        self.use_ocr_lowfeat = opt.use_ocr_lowfeat
        self.use_ocr_ourfeat = opt.use_ocr_ourfeat
        self.use_ocr_with_angle = opt.use_ocr_with_angle
        self.ocr_score_mode = opt.ocr_score_mode
        self.caption_model = opt.caption_model
        assert self.ocr_score_mode in ['normal','rel_same_PTN','rel_diff_PTN','rel2score']

        if self.use_ocr_with_angle:
            self.ocr_box_dim = 8
        else:
            self.ocr_box_dim = 5
        
        if self.use_ocr_lowfeat:
            ocr_feat_dim = self.att_feat_size*2+300+604+self.ocr_box_dim
        elif self.use_ocr_ourfeat:
            ocr_feat_dim = self.att_feat_size+1024+300+604+self.ocr_box_dim
        else:
            ocr_feat_dim = self.att_feat_size+300+604+self.ocr_box_dim

        self.use_bn = opt.use_bn
        self.use_obj_ln = opt.use_obj_ln

        self.ss_prob = 0.0 # Schedule sampling probability

        self.embed = nn.Sequential(nn.Embedding(self.vocab_size + 1, self.input_encoding_size),
                                nn.ReLU(),
                                nn.Dropout(self.drop_prob_lm))
        # self.fc_embed = nn.Sequential(nn.Linear(self.fc_feat_size, self.rnn_size),
        #                             nn.ReLU(),
        #                             nn.Dropout(self.drop_prob_lm))
        if not self.use_obj_ln:
            self.att_embed = nn.Sequential(*(
                                        ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn==1 else ())+
                                        (nn.Linear(self.att_feat_size, self.rnn_size),)+
                                        ((nn.BatchNorm1d(self.rnn_size),) if self.use_bn==2 else ())+
                                        (nn.ReLU(),
                                        nn.Dropout(self.drop_prob_lm))+
                                        ((nn.BatchNorm1d(self.rnn_size),) if self.use_bn==3 else ())))
        else:
            self.att_embed = nn.Sequential(*(
                                    ((LayerNorm(self.att_feat_size),) if self.use_obj_ln==1 else ())+
                                    (nn.Linear(self.att_feat_size, self.rnn_size),)+
                                    ((LayerNorm(self.rnn_size),) if self.use_obj_ln==2 else ())+
                                    (nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm))
                                    ))
        if self.ocr_score_mode in ['rel_same_PTN','rel_diff_PTN']:
            self.ocr_rel_emb = nn.Sequential(*(
                                    (nn.Linear(13, self.input_encoding_size),)+
                                    ((LayerNorm(self.input_encoding_size),) if self.use_obj_ln==2 else ())+
                                    (nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm))
                                    ))
        elif self.ocr_score_mode == 'rel2score':
            self.ocr_rel_emb = nn.Sequential(*(
                                        (nn.Linear(13, 256),)+
                                        ((LayerNorm(256),) if self.use_obj_ln==2 else ())+
                                        (nn.ReLU(),
                                        nn.Dropout(self.drop_prob_lm),
                                        nn.Linear(256, 1),)
                                        ))

        self.logit_layers = getattr(opt, 'logit_layers', 1)
        if self.logit_layers == 1:
            self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)
            self.logit2 = nn.Linear(self.ocr_size, self.ocr_size)
        else:
            self.logit = [[nn.Linear(self.rnn_size, self.rnn_size), nn.ReLU(), nn.Dropout(0.5)] for _ in range(opt.logit_layers - 1)]
            self.logit = nn.Sequential(*(reduce(lambda x,y:x+y, self.logit) + [nn.Linear(self.rnn_size, self.vocab_size + 1)]))
        self.ctx2att = nn.Linear(self.rnn_size, self.att_hid_size)
        if self.ocr_score_mode in ['normal', 'rel2score', 'rel_diff_PTN']:
            self.ocr_ptr_net = OcrPtrNet(self.rnn_size, label_smoothing=self.label_smoothing)
        elif self.ocr_score_mode == 'rel_same_PTN':
            self.ocr_ptr_net = OcrPtrNet(self.rnn_size, self.rnn_size*2, label_smoothing=self.label_smoothing)
        if self.ocr_score_mode == 'rel_diff_PTN':
            self.ocr_ptr_net_rel = OcrPtrNet(self.rnn_size, label_smoothing=self.label_smoothing)

        self.ocr_embed = nn.Sequential(*(
                                    ((LayerNorm(ocr_feat_dim-self.ocr_box_dim),) if self.use_obj_ln==1 else ())+
                                    (nn.Linear(ocr_feat_dim-self.ocr_box_dim, self.rnn_size),)+
                                    ((LayerNorm(self.rnn_size),) if self.use_obj_ln==2 else ())+
                                    (nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm))
                                    ))
        self.ocr_box_embed = nn.Sequential(*(
                                    ((LayerNorm(self.ocr_box_dim),) if self.use_obj_ln==1 else ())+
                                    (nn.Linear(self.ocr_box_dim, self.rnn_size),)+
                                    ((LayerNorm(self.rnn_size),) if self.use_obj_ln==2 else ())+
                                    (nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm))
                                    ))

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers, bsz, self.rnn_size),
                weight.new_zeros(self.num_layers, bsz, self.rnn_size))

    def clip_att(self, att_feats, att_masks):
        # Clip the length of att_masks and att_feats to the maximum length
        if att_masks is not None:
            max_len = att_masks.data.long().sum(1).max()
            att_feats = att_feats[:, :max_len].contiguous()
            att_masks = att_masks[:, :max_len].contiguous()
        return att_feats, att_masks

    def _prepare_feature(self, fc_feats, att_feats, att_masks, ocr_feats, ocr_masks, ocr_relation_feat):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)

        # embed fc and att feats
        # fc_feats = self.fc_embed(fc_feats)
        
        batch_size = att_feats.size(0)
        obj_size = att_feats.size(1)
        att_feats = F.normalize(att_feats, dim=-1)
        if not self.use_obj_ln:
            if self.use_bn:
                att_feats = att_feats.view(-1,self.att_feat_size)
            att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)
            if self.use_bn:
                att_feats = att_feats.view(batch_size, obj_size, -1)
        else:
            # m4c embedding
            att_feats = self.att_embed(att_feats)

        # ocr_feats = self.ocr_embed(ocr_feats[:,:,:-self.ocr_box_dim]) + \
        #             self.ocr_box_embed(ocr_feats[:,:,-self.ocr_box_dim:])

        # m4c embedding
        # att_feats = F.normalize(att_feats, dim=-1)
        # att_feats = self.obj_layernorm(self.obj_embed(att_feats))
        # att_feats = self.obj_drop(att_feats)

        if self.use_ocr_lowfeat:
            aug_visual_start = self.att_feat_size
            aug_visual_end = self.att_feat_size*2
            
        elif self.use_ocr_ourfeat:
            aug_visual_start = self.att_feat_size
            aug_visual_end = self.att_feat_size+1024
        else:
            aug_visual_end = self.att_feat_size

        ocr_visual = F.normalize(ocr_feats[:,:,:self.att_feat_size], dim=-1)
        ocr_fasttext = F.normalize(ocr_feats[:,:,aug_visual_end:aug_visual_end+300], dim=-1)
        ocr_phoc = F.normalize(ocr_feats[:,:,aug_visual_end+300:aug_visual_end+300+604], dim=-1)
        # ocr_visual = ocr_feats[:,:,:self.att_feat_size]
        # ocr_fasttext = ocr_feats[:,:,aug_visual_end:aug_visual_end+300]
        # ocr_phoc = ocr_feats[:,:,aug_visual_end+300:aug_visual_end+300+604]
        if self.use_ocr_lowfeat or self.use_ocr_ourfeat:
            aug_ocr_visual = ocr_feats[:,:,aug_visual_start:aug_visual_end]
            ocr_main_feat = torch.cat(
                [ocr_visual, aug_ocr_visual, ocr_fasttext, ocr_phoc],
                dim=-1
            )
        else:
            ocr_main_feat = torch.cat(
                [ocr_visual, ocr_fasttext, ocr_phoc],
                dim=-1
            )
        # ocr_feats = self.ocr_layernorm(self.ocr_embed(ocr_main_feat)) + \
        #             self.ocr_box_layernorm(self.ocr_box_embed(ocr_feats[:,:,-self.ocr_box_dim:]))
        # ocr_feats = self.ocr_drop(ocr_feats)
        ocr_feats = self.ocr_embed(ocr_main_feat) + self.ocr_box_embed(ocr_feats[:,:,-self.ocr_box_dim:])

        if self.caption_model in ['show_attend_tell_cat', 'topdowncat']:
            if att_masks == None:
                att_masks = torch.ones(att_feats.shape[:2]).cuda()
            att_feats = torch.cat([att_feats, ocr_feats], 1)
            att_masks = torch.cat([att_masks, ocr_masks], 1)

        if self.ocr_score_mode in ['rel_same_PTN','rel_diff_PTN']:
            ocr_rel_feat = self.ocr_rel_emb(ocr_relation_feat)
        elif self.ocr_score_mode == 'rel2score':
            vocab_scores = torch.zeros((att_feats.shape[0], self.vocab_size+1, 1)).expand(-1,-1,self.ocr_size)
            ocr_rel_scores = self.ocr_rel_emb(ocr_relation_feat).squeeze(-1)

            ocr_rel_feat = torch.cat([vocab_scores.cuda(), ocr_rel_scores.view(-1,self.ocr_size,self.ocr_size)], dim=1)
        else:
            ocr_rel_feat = None

        # Project the attention feats first to reduce memory and computation comsumptions.
        p_att_feats = self.ctx2att(att_feats)

        return fc_feats, att_feats, p_att_feats, att_masks, ocr_feats, ocr_masks, ocr_rel_feat

    def _forward(self, ocr_feats, ocr_masks, ocr_relation_feat, fc_feats, att_feats, seq, att_masks=None):
        batch_size = att_feats.size(0)
        state = self.init_hidden(batch_size)

        outputs = att_feats.new_zeros(batch_size, seq.size(1) - 1, self.vocab_size+1+ocr_feats.size(1))

        # Prepare the features
        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, p_ocr_feats, p_ocr_masks, ocr_rel_feat = self._prepare_feature(fc_feats, att_feats, att_masks, ocr_feats, ocr_masks, ocr_relation_feat)
        # pp_att_feats is used for attention, we cache it in advance to reduce computation cost

        if self.caption_model in ['show_attend_tell', 'show_attend_tell_cat']:
            if self.caption_model == 'show_attend_tell_cat':
                state[0][1] = torch.sum(p_att_feats, 1)/torch.sum(p_att_masks, 1).unsqueeze(-1)
            else:
                state[0][1] = p_att_feats.mean(1)

        for i in range(seq.size(1) - 1):
            if self.training and i >= 1 and self.ss_prob > 0.0: # otherwiste no need to sample
                sample_prob = att_feats.new(batch_size).uniform_(0, 1)
                sample_mask = sample_prob < self.ss_prob
                if sample_mask.sum() == 0:
                    it = seq[:, i].clone()
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                    it = seq[:, i].data.clone()
                    #prob_prev = torch.exp(outputs[-1].data.index_select(0, sample_ind)) # fetch prev distribution: shape Nx(M+1)
                    #it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1))
                    # prob_prev = torch.exp(outputs[-1].data) # fetch prev distribution: shape Nx(M+1)
                    prob_prev = torch.exp(outputs[:, i-1].detach()) # fetch prev distribution: shape Nx(M+1)
                    it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
            else:
                it = seq[:, i].clone()          
            # break if all the sequences end
            if i >= 1 and seq[:, i].sum() == 0:
                break

            output, state = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, p_ocr_feats, p_ocr_masks, ocr_rel_feat, state)
            outputs[:, i] = output

        return outputs

    def word_embedding(self, vocab_emb, ocr_emb, it):
        batch_size = it.size(0)
        vocab_num = vocab_emb.size(0)

        assert vocab_emb.size(-1) == ocr_emb.size(-1), "Vocab embedding and ocr embedding do not match."
        vocab_emb = vocab_emb.unsqueeze(0).expand(batch_size, -1, -1)
        vocab_ocr_emb_cat = torch.cat([vocab_emb, ocr_emb], dim=1)
        word_emb = _batch_gather(vocab_ocr_emb_cat, it)

        return word_emb

    def get_logprobs_state(self, it, fc_feats, att_feats, p_att_feats, att_masks, ocr_feats, ocr_masks, ocr_rel_feat, state):
        # 'it' contains a word index
        # xt = self.embed(it)
        xt = self.word_embedding(self.embed[0].weight, ocr_feats, it)

        output, state = self.core(xt, fc_feats, att_feats, p_att_feats, state, att_masks)
        fixed_scores = self.logit(output)  # (b, vocab_size)

        if self.caption_model in ['show_attend_tell', 'show_attend_tell_cat']:
            h_state = state[0][0]
        else:
            h_state = state[0][0]

        if self.ocr_score_mode in ['normal', 'rel2score', 'rel_diff_PTN']:
            dynamic_ocr_scores = self.ocr_ptr_net(h_state, ocr_feats, ocr_masks) # (b, ocr_size)

        if self.ocr_score_mode in ['rel_same_PTN', 'rel_diff_PTN']:
            # rel_feat = _batch_gather(ocr_rel_feat.reshape(ocr_rel_feat.size(0), ocr_rel_feat.size(1), -1), it) 
            # for bi in range(it.size(0)):
            #     if it[bi] <= self.vocab_size:
            #         feat = torch.zeros([1, self.ocr_size, self.rnn_size]).cuda()
            #     else:
            #         feat = ocr_rel_feat[bi,it[bi]-self.vocab_size-1].unsqueeze(0)
            #     if bi == 0:
            #         rel_feat = feat
            #     else:
            #         rel_feat = torch.cat([rel_feat,feat], 0)
            
            # for bi in range(it.size(0)):
            #     if it[bi] > self.vocab_size:
            #         rel_feat[bi] = ocr_rel_feat[bi,it[bi]-self.vocab_size-1].unsqueeze(0)

            ocr_it = it-self.vocab_size-1
            ocr_it = torch.where(ocr_it>=0, ocr_it, torch.ones_like(ocr_it)*self.ocr_size)
            
            zero_pad = torch.zeros([it.size(0), 1, self.ocr_size, self.rnn_size]).cuda()
            all_rel_feat = torch.cat([ocr_rel_feat, zero_pad], 1)
            rel_feat = _batch_gather(all_rel_feat.reshape(all_rel_feat.size(0), all_rel_feat.size(1), -1), ocr_it) 
            rel_feat = rel_feat.reshape(ocr_rel_feat.size(0), ocr_rel_feat.size(2), ocr_rel_feat.size(-1))

            if self.ocr_score_mode == 'rel_same_PTN':
                rel_scores = self.ocr_ptr_net(h_state, torch.cat([ocr_feats, rel_feat], dim=-1), ocr_masks) # (b, ocr_size)
                dynamic_ocr_scores = rel_scores
            elif self.ocr_score_mode == 'rel_diff_PTN':
                rel_scores = self.ocr_ptr_net_rel(h_state, rel_feat, ocr_masks)
                dynamic_ocr_scores = 0.5*dynamic_ocr_scores + 0.5*rel_scores

        if self.ocr_score_mode == 'rel2score':
            rel_scores = _batch_gather(ocr_rel_feat, it)
            dynamic_ocr_scores = 0.5*dynamic_ocr_scores + 0.5*rel_scores

            if self.label_smoothing == 0:
                extended_ocr_mask = (1.0 - ocr_masks) * -1e12
                dynamic_ocr_scores = dynamic_ocr_scores + extended_ocr_mask

        if self.label_smoothing == 0:
            scores = torch.cat([fixed_scores, dynamic_ocr_scores], dim=-1)
            return scores, state
        else:
            dynamic_ocr_scores = self.logit2(dynamic_ocr_scores)
            scores = torch.cat([fixed_scores, dynamic_ocr_scores], dim=-1)

            probs = F.softmax(scores, dim=1)
            logprobs = torch.log(probs)
            fixed_masks = torch.ones_like(fixed_scores)
            both_masks = torch.cat((fixed_masks, ocr_masks), dim = -1)
            both_masks = (1.0 - both_masks) * -10000.0
            logprobs = logprobs + both_masks
            return logprobs, state

    def _sample_beam(self, ocr_feats, ocr_masks, ocr_relation_feat, fc_feats, att_feats, att_masks=None, unk_idx=0, ocrunk_idx=0, opt={}):
        beam_size = opt.get('beam_size', 10)
        batch_size = att_feats.size(0)

        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, p_ocr_feats, p_ocr_masks, ocr_rel_feat = self._prepare_feature(fc_feats, att_feats, att_masks, ocr_feats, ocr_masks, ocr_relation_feat)

        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = self.init_hidden(beam_size)
            tmp_fc_feats = p_fc_feats[k:k+1].expand(beam_size, p_fc_feats.size(1))
            tmp_att_feats = p_att_feats[k:k+1].expand(*((beam_size,)+p_att_feats.size()[1:])).contiguous()
            tmp_p_att_feats = pp_att_feats[k:k+1].expand(*((beam_size,)+pp_att_feats.size()[1:])).contiguous()
            tmp_att_masks = p_att_masks[k:k+1].expand(*((beam_size,)+p_att_masks.size()[1:])).contiguous() if p_att_masks is not None else None
            tmp_ocr_feats = p_ocr_feats[k:k+1].expand(*((beam_size,)+p_ocr_feats.size()[1:])).contiguous()
            tmp_ocr_masks = p_ocr_masks[k:k+1].expand(*((beam_size,)+p_ocr_masks.size()[1:])).contiguous()
            tmp_ocr_rel_feat = ocr_rel_feat[k:k+1].expand(*((beam_size,)+ocr_rel_feat.size()[1:])).contiguous()
            if self.caption_model in ['show_attend_tell', 'show_attend_tell_cat']:
                if self.caption_model == 'show_attend_tell_cat':
                    state[0][1] = torch.sum(tmp_att_feats, 1)/torch.sum(tmp_att_masks, 1).unsqueeze(-1)
                else:
                    state[0][1] = tmp_att_feats.mean(1)

            for t in range(1):
                if t == 0: # input <bos>
                    it = att_feats.new_zeros([beam_size], dtype=torch.long)

                logprobs, state = self.get_logprobs_state(it, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, tmp_att_masks, tmp_ocr_feats, tmp_ocr_masks, tmp_ocr_rel_feat, state)
                logprobs[..., unk_idx] = -1e10
                logprobs[..., ocrunk_idx] = -1e10

            self.done_beams[k] = self.beam_search(unk_idx, ocrunk_idx, state, logprobs, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, tmp_att_masks, tmp_ocr_feats, tmp_ocr_masks, tmp_ocr_rel_feat, opt=opt)
            seq[:, k] = self.done_beams[k][0]['seq'] # the first beam has highest cumulative score
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)

    def _sample(self, ocr_feats, ocr_masks, ocr_relation_feat, fc_feats, att_feats, att_masks=None, unk_idx=0, ocrunk_idx=0, opt={}):

        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        decoding_constraint = opt.get('decoding_constraint', 0)
        if beam_size > 1:
            return self._sample_beam(ocr_feats, ocr_masks, ocr_relation_feat, fc_feats, att_feats, att_masks, unk_idx, ocrunk_idx, opt)

        batch_size = att_feats.size(0)
        state = self.init_hidden(batch_size)

        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, p_ocr_feats, p_ocr_masks, ocr_rel_feat = self._prepare_feature(fc_feats, att_feats, att_masks, ocr_feats, ocr_masks, ocr_relation_feat)
        if self.caption_model in ['show_attend_tell', 'show_attend_tell_cat']:
            if self.caption_model == 'show_attend_tell_cat':
                state[0][1] = torch.sum(p_att_feats, 1)/torch.sum(p_att_masks, 1).unsqueeze(-1)
            else:
                state[0][1] = p_att_feats.mean(1)

        seq = att_feats.new_zeros((batch_size, self.seq_length), dtype=torch.long)
        seqLogprobs = att_feats.new_zeros(batch_size, self.seq_length)
        for t in range(self.seq_length + 1):
            if t == 0: # input <bos>
                it = att_feats.new_zeros(batch_size, dtype=torch.long)

            logprobs, state = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, p_ocr_feats, p_ocr_masks, ocr_rel_feat, state)
            logprobs[..., unk_idx] = -1e10
            logprobs[..., ocrunk_idx] = -1e10
            
            if decoding_constraint and t > 0:
                tmp = logprobs.new_zeros(logprobs.size())
                tmp.scatter_(1, seq[:,t-1].data.unsqueeze(1), float('-inf'))
                logprobs = logprobs + tmp

            # sample the next word
            if t == self.seq_length: # skip if we achieve maximum length
                break
            if sample_max:
                sampleLogprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()
            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(logprobs.data) # fetch prev distribution: shape Nx(M+1)
                else:
                    # scale logprobs by temperature
                    prob_prev = torch.exp(torch.div(logprobs.data, temperature))
                it = torch.multinomial(prob_prev, 1)
                sampleLogprobs = logprobs.gather(1, it) # gather the logprobs at sampled positions
                it = it.view(-1).long() # and flatten indices for downstream processing

            # stop when all finished
            if t == 0:
                unfinished = it > 0
            else:
                unfinished = unfinished * (it > 0)
            it = it * unfinished.type_as(it)
            seq[:,t] = it
            seqLogprobs[:,t] = sampleLogprobs.view(-1)
            # quit loop if all sequences have finished
            if unfinished.sum() == 0:
                break

        return seq, seqLogprobs

class Attention(nn.Module):
    def __init__(self, opt):
        super(Attention, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size

        self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)

    def forward(self, h, att_feats, p_att_feats, att_masks=None):
        # The p_att_feats here is already projected
        att_size = att_feats.numel() // att_feats.size(0) // att_feats.size(-1)
        att = p_att_feats.view(-1, att_size, self.att_hid_size)
        
        att_h = self.h2att(h)                        # batch * att_hid_size
        att_h = att_h.unsqueeze(1).expand_as(att)            # batch * att_size * att_hid_size
        dot = att + att_h                                   # batch * att_size * att_hid_size
        dot = F.tanh(dot)                                # batch * att_size * att_hid_size
        dot = dot.view(-1, self.att_hid_size)               # (batch * att_size) * att_hid_size
        dot = self.alpha_net(dot)                           # (batch * att_size) * 1
        dot = dot.view(-1, att_size)                        # batch * att_size
        
        weight = F.softmax(dot, dim=1)                             # batch * att_size
        if att_masks is not None:
            weight = weight * att_masks.view(-1, att_size).float()
            weight = weight / weight.sum(1, keepdim=True) # normalize to 1
        att_feats_ = att_feats.view(-1, att_size, att_feats.size(-1)) # batch * att_size * att_feat_size
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1) # batch * att_feat_size

        return att_res

class TopDownModel(AttModel):
    def __init__(self, opt):
        super(TopDownModel, self).__init__(opt)
        self.num_layers = 2
        self.core = TopDownCore(opt)

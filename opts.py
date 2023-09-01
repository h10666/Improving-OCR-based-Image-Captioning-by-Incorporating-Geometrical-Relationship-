import argparse

def parse_opt():
    parser = argparse.ArgumentParser()
    # Data input settings
    parser.add_argument('--input_json', type=str, default='data/coco.json',
                    help='path to the json file containing additional info and vocab')
    parser.add_argument('--input_fc_dir', type=str, default='data/cocotalk_fc',
                    help='path to the directory containing the preprocessed fc feats')
    parser.add_argument('--input_att_dir', type=str, default='data/cocotalk_att',
                    help='path to the directory containing the preprocessed att feats')
    parser.add_argument('--input_box_dir', type=str, default='data/cocotalk_box',
                    help='path to the directory containing the boxes of att feats')
    parser.add_argument('--input_ocr_dir', type=str, default='data/m4c_textvqa_ocr_en_frcn_features',
                    help='path to the directory containing the preprocessed att feats')
    parser.add_argument('--input_aug_ocr_dir', type=str, default=None,
                    help='path to the directory containing the preprocessed att feats')
    parser.add_argument('--input_ocr_lowfeat_dir', type=str, default='data/myang_feat/m4c_textvqa_ocr_en_frcn_features_lowfeat',
                    help='path to the directory containing the preprocessed att feats')
    parser.add_argument('--input_ocr_ourfeat_dir', type=str, default='data/myang_feat/m4c_textvqa_ocr_en_spotter_features',
                    help='path to the directory containing the preprocessed att feats')
    parser.add_argument('--input_label_h5', type=str, default='data/coco_label.h5',
                    help='path to the h5file containing the preprocessed dataset')
    parser.add_argument('--start_from', type=str, default=None,
                    help="""continue training from saved model at this path. Path must contain files saved by previous training process: 
                        'infos.pkl'         : configuration;
                        'checkpoint'        : paths to model file(s) (created by tf).
                                              Note: this file contains absolute paths, be careful when moving files around;
                        'model.ckpt-*'      : file(s) with model definition (created by tf)
                    """)
    parser.add_argument('--cached_tokens', type=str, default='coco-train-idxs',
                    help='Cached token file for calculating cider score during self critical training.')

    # Model settings
    parser.add_argument('--fasttext_model_file', type=str, default='../../data/wiki.en.bin',
                    help='')
    parser.add_argument('--caption_model', type=str, default="show_tell",
                    help='show_tell, show_attend_tell, all_img, fc, att2in, att2in2, att2all2, adaatt, adaattmo, topdown, stackatt, denseatt, transformer')
    parser.add_argument('--rnn_size', type=int, default=512,
                    help='size of the rnn in number of hidden nodes in each layer')
    parser.add_argument('--num_layers', type=int, default=1,
                    help='number of layers in the RNN')
    parser.add_argument('--head_num', type=int, default=8,
                    help='number of multi-head')
    parser.add_argument('--rnn_type', type=str, default='lstm',
                    help='rnn, gru, or lstm')
    parser.add_argument('--input_encoding_size', type=int, default=512,
                    help='the encoding size of each token in the vocabulary, and the image.')
    parser.add_argument('--att_hid_size', type=int, default=512,
                    help='the hidden size of the attention MLP; only useful in show_attend_tell; 0 if not using hidden layer')
    parser.add_argument('--fc_feat_size', type=int, default=2048,
                    help='2048 for resnet, 4096 for vgg')
    parser.add_argument('--att_feat_size', type=int, default=2048,
                    help='2048 for resnet, 512 for vgg')
    parser.add_argument('--logit_layers', type=int, default=1,
                    help='number of layers in the RNN')


    parser.add_argument('--use_bn', type=int, default=0,
                    help='If 1, then do batch_normalization before linear layer, if 2 then do bn before relu, if 3 then do bn after relu')
    
    # AoA settings
    parser.add_argument('--mean_feats', type=int, default=1,
                    help='use mean pooling of feats?')
    parser.add_argument('--refine', type=int, default=1,
                    help='refining feature vectors?')
    parser.add_argument('--refine_aoa', type=int, default=1,
                    help='use aoa in the refining module?')
    parser.add_argument('--use_ff', type=int, default=0,
                    help='keep feed-forward layer in the refining module?')
    parser.add_argument('--dropout_aoa', type=float, default=0.3,
                    help='dropout_aoa in the refining module?')

    parser.add_argument('--ctx_drop', type=int, default=1,
                    help='apply dropout to the context vector before fed into LSTM?')
    parser.add_argument('--decoder_type', type=str, default='AoA',
                    help='AoA, LSTM, base')
    parser.add_argument('--use_multi_head', type=int, default=2,
                    help='use multi head attention? 0 for addictive single head; 1 for addictive multi head; 2 for productive multi head.')
    parser.add_argument('--num_heads', type=int, default=8,
                    help='number of attention heads?')
    parser.add_argument('--multi_head_scale', type=int, default=1,
                    help='scale q,k,v?')

    parser.add_argument('--use_warmup', type=int, default=0,
                    help='warm up the learing rate?')
    # parser.add_argument('--acc_steps', type=int, default=1,
    #                 help='accumulation steps')
    
    # by Jing
    parser.add_argument('--epoch_eval_every', type=int, default=4,
                    help='')
    parser.add_argument('--iteration_eval_every', type=int, default=1000000000,
                    help='for debug')
    parser.add_argument('--epoch_test_more', type=int, default=100,
                    help='')
    parser.add_argument('--lstm_layers', type=int, default=1,
                    help='one layer baseline or topdown baseline')
    parser.add_argument('--input_mean_feat', type=int, default=1,
                    help='If input mean_feat to LSTM')
    parser.add_argument('--input_attn_feat', type=int, default=1,
                    help='If input attn_feat to LSTM')
    parser.add_argument('--output_h', type=int, default=1,
                    help='If output h for generating')
    parser.add_argument('--output_attn_feat', type=int, default=0,
                    help='If output attn_feat for generating')
    parser.add_argument('--output_topk_feat', type=int, default=0,
                    help='If output topk_feats for generating')
    parser.add_argument('--output_both', type=int, default=0,
                    help='If output topk_feats and attn_feat for generating')
    parser.add_argument('--k', type=int, default=6,
                    help='choose topk regions')
    parser.add_argument('--use_ocr_lowfeat', type=int, default=0,
                    help='If use features from lower layers for OCR')
    parser.add_argument('--use_ocr_ourfeat', type=int, default=0,
                    help='If use features from our pretrained model for OCR')
    parser.add_argument('--use_ocr_with_angle', type=int, default=0,
                    help='If use OCR tokens from our pretrained model')
    parser.add_argument('--ocr_size', type=int, default=50,
                    help='max ocr token nums')
    parser.add_argument('--use_obj_ln', type=int, default=0,
                    help='If use layernorm for object feature embedding')
    parser.add_argument('--ocr_score_mode', type=str, default='normal',
                    help='normal|rel_same_PTN|rel_diff_PTN|rel2score')

    # feature manipulation
    parser.add_argument('--norm_att_feat', type=int, default=0,
                    help='If normalize attention features')
    parser.add_argument('--use_box', type=int, default=0,
                    help='If use box features')
    parser.add_argument('--norm_box_feat', type=int, default=0,
                    help='If use box, do we normalize box feature')

    # Optimization: General
    parser.add_argument('--max_epochs', type=int, default=-1,
                    help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                    help='minibatch size')
    parser.add_argument('--grad_clip', type=float, default=0.1, #5.,
                    help='clip gradients at this value')
    parser.add_argument('--grad_clip_style', type=str, default='Clamp',
                    help='what clip style to use? Clamp|Norm')
    parser.add_argument('--drop_prob_lm', type=float, default=0.5,
                    help='strength of dropout in the Language Model RNN')
    parser.add_argument('--self_critical_after', type=int, default=-1,
                    help='After what epoch do we start finetuning the CNN? (-1 = disable; never finetune, 0 = finetune from start)')
    parser.add_argument('--seq_per_img', type=int, default=5,
                    help='number of captions to sample for each image during training. Done for efficiency since CNN forward pass is expensive. E.g. coco has 5 sents/image')
    parser.add_argument('--beam_size', type=int, default=1,
                    help='used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')

    #Optimization: for the Language Model
    parser.add_argument('--optim', type=str, default='adam',
                    help='what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
    parser.add_argument('--learning_rate', type=float, default=4e-4,
                    help='learning rate')
    parser.add_argument('--learning_rate_decay_start', type=int, default=-1, 
                    help='at what iteration to start decaying learning rate? (-1 = dont) (in epoch)')
    parser.add_argument('--learning_rate_decay_every', type=int, default=3, 
                    help='every how many iterations thereafter to drop LR?(in epoch)')
    parser.add_argument('--learning_rate_decay_rate', type=float, default=0.8, 
                    help='every how many iterations thereafter to drop LR?(in epoch)')
    parser.add_argument('--optim_alpha', type=float, default=0.9,
                    help='alpha for adam')
    parser.add_argument('--optim_beta', type=float, default=0.999,
                    help='beta used for adam')
    parser.add_argument('--optim_epsilon', type=float, default=1e-8,
                    help='epsilon that goes into denominator for smoothing')
    parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight_decay')

    parser.add_argument('--scheduled_sampling_start', type=int, default=-1, 
                    help='at what iteration to start decay gt probability')
    parser.add_argument('--scheduled_sampling_increase_every', type=int, default=5, 
                    help='every how many iterations thereafter to gt probability')
    parser.add_argument('--scheduled_sampling_increase_prob', type=float, default=0.05, 
                    help='How much to update the prob')
    parser.add_argument('--scheduled_sampling_max_prob', type=float, default=0.25, 
                    help='Maximum scheduled sampling prob.')


    # Evaluation/Checkpointing
    parser.add_argument('--val_images_use', type=int, default=3200,
                    help='how many images to use when periodically evaluating the validation loss? (-1 = all)')
    parser.add_argument('--save_checkpoint_every', type=int, default=2500,
                    help='how often to save a model checkpoint (in iterations)?')
    parser.add_argument('--checkpoint_path', type=str, default='save',
                    help='directory to store checkpointed models')
    parser.add_argument('--language_eval', type=int, default=0,
                    help='Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
    parser.add_argument('--losses_log_every', type=int, default=25,
                    help='How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')       
    parser.add_argument('--load_best_score', type=int, default=1,
                    help='Do we load previous best score when resuming training.')  

    parser.add_argument('--result_dir', type=str, default='results',
                    help='directory to store all the results')     

    parser.add_argument('--print_every', type=int, default=20,
                    help='how often to print logs (in iterations)?')

    # misc
    parser.add_argument('--id', type=str, default='',
                    help='an id identifying this run/job. used in cross-val and appended when writing progress files')
    parser.add_argument('--train_only', type=int, default=0,
                    help='if true then use 80k, else use 110k')


    # Reward
    parser.add_argument('--cider_reward_weight', type=float, default=1,
                    help='The reward weight from cider')
    parser.add_argument('--bleu_reward_weight', type=float, default=0,
                    help='The reward weight from bleu4')

    # Transformer
    parser.add_argument('--label_smoothing', type=float, default=0,
                    help='')
    parser.add_argument('--noamopt', action='store_true',
                    help='')
    parser.add_argument('--noamopt_warmup', type=int, default=2000,
                    help='')
    parser.add_argument('--noamopt_factor', type=float, default=1,
                    help='')

    parser.add_argument('--reduce_on_plateau', action='store_true',
                    help='')

    args = parser.parse_args()

    # Check if args are valid
    assert args.rnn_size > 0, "rnn_size should be greater than 0"
    assert args.num_layers > 0, "num_layers should be greater than 0"
    assert args.input_encoding_size > 0, "input_encoding_size should be greater than 0"
    assert args.batch_size > 0, "batch_size should be greater than 0"
    assert args.drop_prob_lm >= 0 and args.drop_prob_lm < 1, "drop_prob_lm should be between 0 and 1"
    assert args.seq_per_img > 0, "seq_per_img should be greater than 0"
    assert args.beam_size > 0, "beam_size should be greater than 0"
    assert args.save_checkpoint_every > 0, "save_checkpoint_every should be greater than 0"
    assert args.losses_log_every > 0, "losses_log_every should be greater than 0"
    assert args.language_eval == 0 or args.language_eval == 1, "language_eval should be 0 or 1"
    assert args.load_best_score == 0 or args.load_best_score == 1, "language_eval should be 0 or 1"
    assert args.train_only == 0 or args.train_only == 1, "language_eval should be 0 or 1"

    return args

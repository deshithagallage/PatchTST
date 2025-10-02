"""
Enhanced PatchTST Experiment Script with AutoCorrelation and Fourier Cross-Attention
This script shows how to run experiments with the enhanced attention mechanisms.
"""
import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np

def add_enhanced_attention_args(parser):
    """Add enhanced attention mechanism arguments"""
    
    # Enhanced attention mechanisms
    parser.add_argument('--use_autocorr', action='store_true', default=False,
                        help='Enable AutoCorrelation attention for seasonal patterns')
    parser.add_argument('--use_fourier', action='store_true', default=False,
                        help='Enable Fourier Cross-Attention for frequency patterns')
    parser.add_argument('--use_channel_attention', action='store_true', default=False,
                        help='Enable Channel attention mechanism')
    parser.add_argument('--use_temporal_attention', action='store_true', default=False,
                        help='Enable Temporal attention for long-range dependencies')
    
    # AutoCorrelation specific parameters
    parser.add_argument('--autocorr_factor', type=int, default=1,
                        help='Factor for AutoCorrelation period discovery')
    
    # Fourier Cross-Attention specific parameters
    parser.add_argument('--fourier_modes', type=int, default=32,
                        help='Number of Fourier modes for frequency analysis')
    
    # Channel attention parameters
    parser.add_argument('--channel_reduction', type=int, default=16,
                        help='Reduction ratio for channel attention')
    
    return parser

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Enhanced PatchTST with Advanced Attention Mechanisms')

    # random seed
    parser.add_argument('--random_seed', type=int, default=2021, help='random seed')

    # basic config
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='PatchTST',
                        help='model name, options: [PatchTST]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=336, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

    # PatchTST
    parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
    parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--padding_patch', type=str, default='end', help='None: None; end: padding on the end')
    parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
    parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
    parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
    parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
    parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
    parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

    # model define
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=128, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=16, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=256, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', default=True, help='whether to use distilling in encoder, using this argument means not using distilling', action='store_false')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start for OneCycleLR scheduler')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # Add enhanced attention arguments
    parser = add_enhanced_attention_args(parser)

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128], help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    args = parser.parse_args()

    # Set up logging directory
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    if not os.path.exists('./logs/LongForecasting'):
        os.makedirs('./logs/LongForecasting')

    # random seed
    fix_seed = args.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            exp = Exp_Main(args)  # set experiments
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.fc_dropout,
                args.embed,
                args.distil,
                'Exp')
            
            # Add enhanced attention suffixes to setting name
            enhanced_suffix = []
            if args.use_autocorr:
                enhanced_suffix.append('AC')
            if args.use_fourier:
                enhanced_suffix.append('FC')
            if args.use_channel_attention:
                enhanced_suffix.append('CA')
            if args.use_temporal_attention:
                enhanced_suffix.append('TA')
            
            if enhanced_suffix:
                setting += '_' + '_'.join(enhanced_suffix)
            
            setting += '_{}'.format(args.des)
            setting += '_{}'.format(ii)

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.fc_dropout,
            args.embed,
            args.distil,
            'Exp')
        
        # Add enhanced attention suffixes to setting name
        enhanced_suffix = []
        if args.use_autocorr:
            enhanced_suffix.append('AC')
        if args.use_fourier:
            enhanced_suffix.append('FC')
        if args.use_channel_attention:
            enhanced_suffix.append('CA')
        if args.use_temporal_attention:
            enhanced_suffix.append('TA')
        
        if enhanced_suffix:
            setting += '_' + '_'.join(enhanced_suffix)
        
        setting += '_{}'.format(args.des)
        setting += '_{}'.format(ii)

        exp = Exp_Main(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
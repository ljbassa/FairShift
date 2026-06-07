import math
import argparse
import torch
import torch.nn.functional as F
import torch.nn as nn
from diffusion.diffusion_base import cosine_beta_schedule, linear_beta_schedule, Tt1_beta_schedule
from diffusion.diffusion_binomial_active import BinomialDiffusionActive
from layers.layers import TGNN_degree_guided
from functools import partial

def add_model_args(parser):
    # Model params
    parser.add_argument('--loss_type', type=str, default='vb_kl')
    parser.add_argument('--diffusion_steps', type=int, default=1000)
    parser.add_argument('--diffusion_dim', type=int, default=64)
    parser.add_argument('--dp_rate', type=float, default=0.)
    parser.add_argument('--edge_dropout', type=float, default=0.0,
                        help='drop edges in message passing during training (0.0 disables)')
    parser.add_argument('--num_heads', type=int, nargs="*", default=[8, 8, 8, 8, 1])
    parser.add_argument('--final_prob_node', type=float, nargs="*", default=None)
    parser.add_argument('--final_prob_edge', type=float, nargs="*", default=[1-1e-12, 1e-12])
    parser.add_argument('--parametrization', type=str, default='xt_prescribed_st')
    parser.add_argument('--sample_time_method', type=str, default='importance')
    parser.add_argument('--noise_schedule', type=str, default='cosine', help='cosine | linear')
    parser.add_argument('--norm', type=str, default='None', help='None | BN' )
    parser.add_argument('--use_node_feat', action='store_true', help='inject pyg_data.x into MPB')
    parser.add_argument('--predict_s', action='store_true')
    parser.add_argument('--active_method', type=str, default='topk', choices=['topk','threshold','bernoulli'])
    parser.add_argument('--active_ratio', type=float, default=0.05)
    parser.add_argument('--active_threshold', type=float, default=0.5)
    parser.add_argument('--s_loss_weight', type=float, default=1.0)
    parser.add_argument('--ratio_loss_weight', type=float, default=0.1)
    parser.add_argument('--s_pos_weight_cap', type=float, default=50.0)
    parser.add_argument('--fair_score_sp', action='store_true')
    parser.add_argument('--fair_score_eta', type=float, default=0.0)
    parser.add_argument('--fair_score_k', type=float, default=0.15)
    parser.add_argument('--fair_score_apply_sample', type=eval, default=True)
    parser.add_argument('--fair_score_guidance_normalize', type=eval, default=False)
    parser.add_argument('--fair_label_attr', type=str, default='y')
    parser.add_argument('--fair_sensitive_attr', type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--fair_sensitive_value', type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--fair_edge_sensitive_mode', type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--node_feat_dropout', type=float, default=0.0)
    parser.add_argument('--node_feat_mask_prob', type=float, default=0.0)
    parser.add_argument('--degree_t_jitter', type=int, default=0)
    parser.add_argument('--degree_0_jitter', type=int, default=0)
    parser.add_argument('--degree_t_mask_prob', type=float, default=0.0)
    parser.add_argument('--degree_0_mask_prob', type=float, default=0.0)
    parser.add_argument('--global_context_dropout', type=float, default=0.0)


def get_model_id(args):
    return 'multinomial_diffusion'

def get_model(args, initial_graph_sampler):
    if args.final_prob_node is not None:
        assert sum(args.final_prob_node) == 1
        assert len(args.final_prob_node) == args.num_node_classes
    assert sum(args.final_prob_edge) == 1
    assert len(args.final_prob_edge) == args.num_edge_classes

    # Always use TGNN_degree_guided and BinomialDiffusionActive
    dynamics_fn = TGNN_degree_guided
    diffusion_fn = BinomialDiffusionActive
        
    dynamics = dynamics_fn(
            max_degree=args.max_degree,
            num_node_classes=2 if args.num_node_classes is None else args.num_node_classes, 
            num_edge_classes=args.num_edge_classes,
            dim=args.diffusion_dim,
            num_steps=args.diffusion_steps,
            num_heads=args.num_heads,
            dropout=args.dp_rate,
            edge_dropout=args.edge_dropout,
            norm=args.norm,
            gru=True,
            degree=args.degree,
            augmented_features=args.augmented_feature_dict,
            return_node_class = args.has_node_feature,
            use_node_feat=args.use_node_feat,
            predict_s=args.predict_s,
            node_feat_dropout=args.node_feat_dropout,
            node_feat_mask_prob=args.node_feat_mask_prob,
            degree_t_jitter=args.degree_t_jitter,
            degree_0_jitter=args.degree_0_jitter,
            degree_t_mask_prob=args.degree_t_mask_prob,
            degree_0_mask_prob=args.degree_0_mask_prob,
            global_context_dropout=args.global_context_dropout,
    )

    if args.noise_schedule == 'cosine':
        noise_schedule = cosine_beta_schedule
    elif args.noise_schedule == 'linear':
        noise_schedule = linear_beta_schedule
    elif args.noise_schedule == 'Tt1':
        noise_schedule = Tt1_beta_schedule
    else:
        raise NotImplementedError()

    base_dist = diffusion_fn(
        args.num_node_classes, args.num_edge_classes, initial_graph_sampler, dynamics, timesteps=args.diffusion_steps, 
        loss_type=args.loss_type, final_prob_node=args.final_prob_node, final_prob_edge=args.final_prob_edge,
        parametrization=args.parametrization, sample_time_method=args.sample_time_method,
        noise_schedule=noise_schedule, device=args.device,
        # [ADD] predict_s
        predict_s=args.predict_s,
        active_method=args.active_method,
        active_ratio=args.active_ratio,
        active_threshold=args.active_threshold,
        s_loss_weight=args.s_loss_weight,
        ratio_loss_weight=args.ratio_loss_weight,
        s_pos_weight_cap=args.s_pos_weight_cap,
        fair_score_sp=getattr(args, 'fair_score_sp', False),
        fair_score_eta=getattr(args, 'fair_score_eta', 0.0),
        fair_score_k=getattr(args, 'fair_score_k', 0.15),
        fair_score_apply_sample=getattr(args, 'fair_score_apply_sample', True),
        fair_score_guidance_normalize=getattr(args, 'fair_score_guidance_normalize', False),
        fair_label_attr=getattr(args, 'fair_label_attr', getattr(args, 'fair_sensitive_attr', 'y')),
        )

    return base_dist

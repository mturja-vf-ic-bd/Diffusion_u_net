def get_argparser():
    import argparse
    parser = argparse.ArgumentParser(description='Input training parameters')
    parser.add_argument('--fold', type=int, default=0,
                        help='fold number')
    parser.add_argument('--write_dir', type=str,
                        help="Output directory name")
    parser.add_argument('--batch_size', type=int, default=32, nargs='?',
                        help="Output directory name")
    parser.add_argument('--lr', type=float, default=1e-2, nargs='?',
                        help="Output directory name")
    parser.add_argument('--max_epoch', type=int, default=5000, nargs='?')
    parser.add_argument('--split', type=int, default=500, nargs='?')
    parser.add_argument('--n_cls_lyr', type=int, default=3, nargs='?')
    parser.add_argument('--cls_hdn', type=int, default=256, nargs='?')
    parser.add_argument('--n_attn_lyr', type=int, default=2, nargs='?')
    parser.add_argument('--n', type=int, default=148, nargs='?')
    parser.add_argument('--attn_hdn', type=int, default=32, nargs='?')
    parser.add_argument('--n_neigh', type=int, default=2, nargs='?')
    parser.add_argument('--train_r', type=float, default=.7, nargs='?')
    parser.add_argument('--valid_r', type=float, default=.2, nargs='?')
    parser.add_argument('--dropout', type=float, default=0.2, nargs='?')
    parser.add_argument('--loss_w', type=float, default=[1, 5, 0.001, 0.05], nargs='+')
    parser.add_argument('--train', type=bool, default=False)
    parser.add_argument('--ks', type=float, default=[0.7, 0.7, 0.7, 0.7, 0.7, 0.7], nargs='+')
    parser.add_argument('--comment', type=str, default="No Comment")
    parser.add_argument('--indim', type=int, default=8)
    parser.add_argument('--common_unet', type=bool, default=False)
    parser.add_argument('--modes', type=int, default=10)
    parser.add_argument('--kernel', type=int, default=16)
    parser.add_argument('--opt', type=str, default="Adam")
    parser.add_argument('--model_ckp_path', type=str)
    parser.add_argument('--load', type=bool, default=False)
    parser.add_argument('--prev_model', type=str)
    return parser
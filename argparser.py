import argparse
from argparse import ArgumentParser


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    parser = ArgumentParser(description="LanguageProcessors Arguments")
    parser.add_argument('--model', type=str, default="Transformer",
                        choices=["Transformer",
                                 "SharedTransformer",
                                 "SharedGatedTransformer",
                                 "UT",
                                 "UT_end",
                                 "GUT",
                                 "GUTX_end",
                                 "GUT_end",
                                 "GUT_token_end",
                                 "GUT_nogate_end",
                                 "GUT_notrans_end",
                                 "TLB",
                                 "TLB10",
                                 "GUTLB",
                                 "TLB100",
                                 "GUTLB100"])
    parser.add_argument('--no_display', type=str2bool, default=False, const=True, nargs='?')
    parser.add_argument('--display_params', type=str2bool, default=True, const=True, nargs='?')
    parser.add_argument('--test', type=str2bool, default=False, const=True, nargs='?')
    parser.add_argument('--model_type', type=str, default="classifier",
                        choices=["sentence_pair", "classifier", "flipflop", "seqlabel"])
    parser.add_argument('--dataset', type=str, default="IMDB_lra",
                        choices=[="listopsc2", "listops_lra", "IMDB_lra", "AAN_lra", "cifar10_lra", "pathfinder_lra",
                                 "cifar10_lra_sparse", "pathfinder_lra_sparse", "proplogic", "flipflop"])
    parser.add_argument('--times', type=int, default=3)
    parser.add_argument('--initial_time', type=int, default=0)
    parser.add_argument('--truncate_k', type=str2bool, default=False, const=True, nargs='?')
    parser.add_argument('--limit', type=int, default=-1)
    parser.add_argument('--display_step', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--example_display_step', type=int, default=500)
    parser.add_argument('--load_checkpoint', type=str2bool, default=False, const=True, nargs='?')
    parser.add_argument('--reproducible', type=str2bool, default=True, const=True, nargs='?')
    return parser

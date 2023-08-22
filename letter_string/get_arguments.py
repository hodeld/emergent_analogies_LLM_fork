# Settings
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--sentence', action='store_true', help="Present problem in sentence format.", default=False)
parser.add_argument('--noprompt', action='store_true', help="Present problem without prompt.", default=False)
parser.add_argument('--replica', action='store_true', help="Replicate results", default=False)
parser.add_argument('--subset', action='store_true', help="Use subset", default=True)
parser.add_argument('--no-subset', dest='subset', action='store_false', help="Do not use subset")
parser.add_argument('--modified', action='store_true', help="Use modified problems", default=True)
parser.add_argument('--no-modified', dest='modified', action='store_false', help="Do not use modified problems")
parser.add_argument('--synthetic', action='store_true', help="Use synthetic alphabet", default=True)
parser.add_argument('--no-synthetic', dest='synthetic', action='store_false', help="Do not use synthetic alphabet")


args = parser.parse_args()

def get_suffix_modified():
    if args.synthetic:
        suffix = '_modified_synthetic'
    else:
        suffix = '_modified'
    return suffix

def get_suffix(args):
    save_fname = ''
    if args.sentence:
        save_fname += '_sentence'
    if args.noprompt:
        save_fname += '_noprompt'

    if args.modified:
        save_fname += get_suffix_modified()
    elif args.replica:
        save_fname += '_replica'
    if args.subset:
        save_fname += '_subset'

    return save_fname


def get_suffix_problems(args):
    save_fname = ''
    if args.modified:
        save_fname += get_suffix_modified()
    return save_fname


def get_prob_types(args, all_prob, prob_types):
    if args.subset and not args.modified:  # todo get subset from modified
        pass  # prob_types = ['succ'] # define subset
    return prob_types
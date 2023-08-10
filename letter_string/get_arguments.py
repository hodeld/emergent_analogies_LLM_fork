# Settings
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--sentence', action='store_true', help="Present problem in sentence format.", default=False)
parser.add_argument('--noprompt', action='store_true', help="Present problem without prompt.", default=False)
parser.add_argument('--replica', action='store_true', help="Replicate results", default=False)
parser.add_argument('--subset', action='store_true', help="Replicate results", default=True)
parser.add_argument('--modified', action='store_true', help="Replicate results", default=True)

args = parser.parse_args()


def get_suffix(args):
    save_fname = ''
    if args.sentence:
        save_fname += '_sentence'
    if args.noprompt:
        save_fname += '_noprompt'

    if args.modified:
        save_fname += '_modified'
    elif args.replica:
        save_fname += '_replica'
    if args.subset:
        save_fname += '_subset'

    return save_fname


def get_suffix_problems(args):
    save_fname = ''
    if args.modified:
        save_fname += '_modified'
    return save_fname
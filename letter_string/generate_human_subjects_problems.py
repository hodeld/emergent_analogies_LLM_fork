import json
import random
import shutil

import openai
import numpy as np
import builtins
import argparse
import os
import time

import pandas as pd

from get_arguments import args, get_suffix, get_suffix_problems, get_prob_types, get_version_dir
from letter_string.eval_GPT3_letterstring_prob import generate_prompt
from letter_string.gen_problems import save_prob

CONDITIONS = ['original', 'modified', 'modified_synthetic']


def generate_problems_js(args):

	"""creates problems from different versions to use in online experiment"""
	# Load all problems
	conditions = CONDITIONS
	prob_types = ['succ', 'pred', 'add_letter', 'remove_redundant', 'fix_alphabet', 'sort']
	all_prob_js = {}
	for k, cond in enumerate(conditions):
		suffix = f'_{cond}'
		all_prob = np.load(f'./all_prob{suffix}.npz', allow_pickle=True)['all_prob']
		all_prob_dict = all_prob.item()
		#prob_types = builtins.list(all_prob.item().keys())
		#prob_types = get_prob_types(args, all_prob, prob_types)
		all_prob_types = prob_types
		# create
		if 'synthetic' in cond:
			args.synthetic, args.alphabetprompt = True, True
		else: # elif 'synthetic' not in cond: or 'original' in cond
			args.synthetic, args.alphabetprompt= False, False

		for prob_type in all_prob_types:
			probs = all_prob_dict[prob_type]
			prob_name = f'{cond}_{prob_type}'
			all_prob_js = save_prob(probs, prob_name, all_prob_js, args)

	all_prob_json_string = json.dumps(all_prob_js)
	# Write to js script
	js_fname = f'letterstring_online_experiment/all_prob_humansubjects.js'
	js_fid = open(js_fname, 'w')
	js_fid.write('var all_problems = ' + all_prob_json_string)
	js_fid.close()


def create_codes():
	def create_string(r=None):
		generated_string = ''.join(random.choices(sample_str, k=length_of_string))
		return generated_string

	sample_str = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
	length_of_string = 6
	df = pd.DataFrame(range(1000), columns=['id'])
	js_fname = f'letterstring_online_experiment/codes.js'
	df['code'] = df.apply(create_string, axis=1)
	codes_js_string = df['code'].to_json()
	#result = df.to_json(orient="split")
	with open(js_fname, 'w') as jsf:
		jsf.write('var codes = ' + codes_js_string)
	pd.read_json()


if __name__ == '__main__':
	create_codes()
	#generate_problems_js(args)
import shutil

import openai
import numpy as np
import builtins
import argparse
import os
import time

from get_arguments import args, get_suffix, get_suffix_problems, get_prob_types
from secrets_openai import OPENAI_KEY



def generate_prompt(prob, args):

	prompt = ''
	if not args.noprompt:
		prompt += "Let's try to complete the pattern:\n\n"
	if args.sentence:
		prompt += 'If '
		for i in range(len(prob[0][0])):
			prompt += str(prob[0][0][i])
			if i < len(prob[0][0]) - 1:
				prompt += ' '
		prompt += ' changes to '
		for i in range(len(prob[0][1])):
			prompt += str(prob[0][1][i])
			if i < len(prob[0][1]) - 1:
				prompt += ' '
		prompt += ', then '
		for i in range(len(prob[1][0])):
			prompt += str(prob[1][0][i])
			if i < len(prob[1][0]) - 1:
				prompt += ' '
		prompt += ' should change to '
	else:
		prompt += '['
		for i in range(len(prob[0][0])):
			prompt += str(prob[0][0][i])
			if i < len(prob[0][0]) - 1:
				prompt += ' '
		prompt += '] ['
		for i in range(len(prob[0][1])):
			prompt += str(prob[0][1][i])
			if i < len(prob[0][1]) - 1:
				prompt += ' '
		prompt += ']\n['
		for i in range(len(prob[1][0])):
			prompt += str(prob[1][0][i])
			if i < len(prob[1][0]) - 1:
				prompt += ' '
		prompt += '] ['
	return prompt

def main(args):
	# GPT-3 settings
	openai.api_key = OPENAI_KEY
	if args.sentence:
		kwargs = { "engine":"text-davinci-003", "temperature":0, "max_tokens":40, "echo":False, "logprobs":1, }
	else:
		kwargs = { "engine":"text-davinci-003", "temperature":0, "max_tokens":40, "stop":"\n", "echo":False, "logprobs":1, }

	# Load all problems
	suffix = get_suffix_problems(args)
	all_prob = np.load(f'./all_prob{suffix}.npz', allow_pickle=True)['all_prob']
	prob_types = builtins.list(all_prob.item().keys())
	prob_types = get_prob_types(args, all_prob, prob_types)
	N_prob_types = len(prob_types)
	new_file = True
	# Evaluate
	N_trials_per_prob_type = 50
	print('N_trials_per_prob_type', N_trials_per_prob_type)
	all_prob_type_responses = []
	for p in range(N_prob_types):
		print('problem type' + str(p+1) + ' of ' + str(N_prob_types) + '...')
		prob_type_responses = []
		for t in range(N_trials_per_prob_type):
			print('trial ' + str(t+1) + ' of ' + str(N_trials_per_prob_type) + '...')
			# Generate prompt
			prob = all_prob.item()[prob_types[p]]['prob'][t]
			prompt = generate_prompt(prob, args)
			# Get response
			response = []
			while len(response) == 0:
				try:
					response = openai.Completion.create(prompt=prompt, **kwargs) # {'choices': [{'text' : 'test'}]} #
				except:
					print('trying again...')
					time.sleep(5)
			prob_type_responses.append(response['choices'][0]['text'])
			if t == 1:
				print('prob', prob, prompt, response['choices'][0]['text'])
		all_prob_type_responses.append(prob_type_responses)
		# Save
		save_fname = 'gpt3_responses/gpt3_letterstring_results'
		save_fname += get_suffix(args)
		save_fname += '.npz'
		if os.path.isfile(save_fname) and new_file:
			new_name = save_fname.replace('.npz', '_copy.npz')
			shutil.copyfile(save_fname, new_name)
			new_file = False
		np.savez(save_fname, all_prob_type_responses=all_prob_type_responses, allow_pickle=True)


if __name__ == '__main__':
	main(args)
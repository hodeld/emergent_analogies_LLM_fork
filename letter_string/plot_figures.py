import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.stats.proportion import proportion_confint

from letter_string.analyze_gpt3_letterstring import TRANS, hide_top_right, get_problems
from letter_string.eval_human_subjects import COL_CONDITION, COL_PROB_TYPE, COL_ACC
from letter_string.generate_human_subjects_problems import CONDITIONS
from letter_string.get_arguments import get_prob_types, args

COL_ALL_ACC = 'all_acc'
COL_ALL_ERR = 'all_err'
RESULTS_DIR = 'human_vs_GPT3/'
modified_synthetic = 'Interval & synthetic alphabet'
modified = 'Interval'

COLOR_DIC_H = {'original_study': '#abd9e9', 'original': '#2c7bb6', 'modified': '#fdae61', 'modified_synthetic': '#d7191c'}
LEGEND_DIC_H = {'original_study': 'Original study', 'original': 'Original', 'modified': modified,
			'modified_synthetic': modified_synthetic}
COLOR_DIC_GPT3 = {'0_original': '#2c7bb6', '1_real': '#fdae61', '2_real_prompt': '#abd9e9', '3_synthetic': '#d7191c'}

LEGEND_DIC_GPT3 = {'0_original': 'Original', '1_real': modified, '2_real_prompt': 'Original & prompt',
				'3_synthetic': modified_synthetic, '4_synthetic_interval1': 'Original & synthetic alphabet'}
COlORS_3 = ['#8da0cb',  '#66c2a5', '#fc8d62',]
COLORS_TWO = ['#8da0cb', '#fc8d62']
LEGENDS_TWO = ['GPT-3', 'Human']


def get_results_versions(versions):
	modified_versions = ['1_real', '2_real_prompt', '3_synthetic', '4_synthetic_interval1']
	all_versions = ['0_original'] + modified_versions
	modified_version_dirs = [os.path.join('GPT3_results_modified_versions', vrs) for vrs in modified_versions]
	gpt3_origin_dir = './GPT3_results'
	all_version_dirs = [gpt3_origin_dir] + modified_version_dirs
	versions_dir_d = {k: v for k, v in zip(all_versions, all_version_dirs)}
	version_dirs = [versions_dir_d[k] for k in versions] #[v for k, v in versions_dir_d.items() if k in versions]
	# Plot settings

	#colors = [v for k, v in COLOR_DIC_GPT3.items() if k in versions]
	colors = COlORS_3[:len(versions)]
	legend_li = [LEGEND_DIC_GPT3[k] for k in versions]
	versions_fps = [vrs_dir + '/zerogen_acc.npz' for vrs_dir in version_dirs]
	results_dicts = [np.load(fp) for fp in versions_fps]
	return results_dicts, colors, legend_li


def plot_paper_comparison(prob_types, versions):

	results_dir = RESULTS_DIR #'GPT3_results_modified_versions/'
	fp = results_dir + 'zerogen_acc_comparison_versions.png'
	results_dicts, colors, legend_li = get_results_versions(versions)
	do_plot(results_dicts, colors, legend_li, prob_types, fp)


def do_plot(results_dicts, colors, legend_li, prob_types=TRANS, fp='zerogen_acc_comparison_versions.png',
			title='GPT-3 zero-generalization problems'):
	all_zerogen_prob_type_names = ['Successor', 'Predecessor', 'Extend\nsequence', 'Remove\nredundant\nletter',
								   'Fix\nalphabetic\nsequence', 'Sort']
	all_zerogen_prob_type_names = pd.Series(all_zerogen_prob_type_names, index=TRANS).loc[prob_types].values
	prob_types_s = pd.Series(range(len(prob_types)), index=prob_types)
	trans_order = ['add_letter', 'succ', 'pred', 'remove_redundant', 'fix_alphabet', 'sort']  # according to the paper
	rank_order = prob_types_s.loc[trans_order].values

	plot_fontsize = 10
	title_fontsize = 12
	axis_label_fontsize = 12
	nr_bars = len(results_dicts)
	bar_width = 0.8
	ind_bar_width = bar_width / nr_bars

	## Zero-generalization problems, grouped by transformation type
	# Load results

	x_points = np.arange(len(all_zerogen_prob_type_names))
	x_points_i = x_points - ((nr_bars - 1)/2 * ind_bar_width)
	ax = plt.subplot(111)
	for i, (result_dict, color_i) in enumerate(zip(results_dicts, colors)):
		zerogen_acc = result_dict['all_acc']
		zerogen_err = result_dict['all_err']
		xvals, yvals = x_points_i + i*(ind_bar_width), zerogen_acc[rank_order]
		bar_container = plt.bar(xvals, yvals, yerr=zerogen_err[:, rank_order],
				color=color_i, edgecolor='black', width=ind_bar_width, ecolor='gray')
		label_fmt = '{:,.2f}'
		if nr_bars > 2 and i == 3:
			ax.bar_label(bar_container, fmt=label_fmt) #, rotation=30)

	plt.ylim([0, 1])
	plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1], ['0', '0.2', '0.4', '0.6', '0.8', '1'], fontsize=plot_fontsize)
	plt.ylabel('Generative accuracy', fontsize=axis_label_fontsize)
	plt.xticks(x_points, np.array(all_zerogen_prob_type_names)[rank_order], fontsize=plot_fontsize)
	plt.xlabel('Transformation type', fontsize=axis_label_fontsize)
	#plt.title(title, pad=40)
	plt.legend(legend_li, fontsize=10, ncol=2, bbox_to_anchor=(1, 1.15)) # frameon=False, loc='upper right')
	hide_top_right(ax)
	plt.tight_layout()
	plt.savefig(fp, dpi=300, bbox_inches="tight")
	plt.show()
	plt.close()


def plot_modified_gpt3():
	conditions_mod = ['0_original', '1_real',  '3_synthetic']
	conditions_comprehension = ['0_original', '4_synthetic_interval1', '2_real_prompt']
	all_prob = get_problems(os.path.join('GPT3_results_modified_versions', '1_real'))
	prob_types = list(all_prob.item().keys())
	prob_types = get_prob_types(args, all_prob, prob_types)
	#versions = ('2_real_prompt', '3_synthetic')
	results_dir = RESULTS_DIR  # 'GPT3_results_modified_versions/'
	plots = ['counterexamples', 'comprehension']
	cond_lists = [conditions_mod, conditions_comprehension]
	for plot_name, conditions in zip(plots, cond_lists):
		fp = results_dir + f'gpt3_zerogen_acc_comparison_{plot_name}.png'
		results_dicts, colors, legend_li = get_results_versions(conditions)
		do_plot(results_dicts, colors, legend_li, prob_types, fp, plot_name.title())


def get_colors_legend(conditions):
	#colors = [v for k, v in COLOR_DIC_H.items() if k in conditions]
	colors = COlORS_3[:len(conditions)]
	legend_li = [v for k, v in LEGEND_DIC_H.items() if k in conditions]
	return colors, legend_li


def plot_humansubjects_uw(conditions=CONDITIONS):
	def adjust_order(r_dict, trans_order_i):
		trans_order_s = pd.Series(range(len(trans_order_i)), index=trans_order_i)
		rank_order = trans_order_s.loc[prob_types].values

		r_dict[COL_ALL_ACC] = pd.Series(r_dict[COL_ALL_ACC]).loc[rank_order].values
		r_dict[COL_ALL_ERR] = np.array([pd.Series(r_dict[COL_ALL_ERR][0]).loc[rank_order].values,
									  pd.Series(r_dict[COL_ALL_ERR][1]).loc[rank_order].values])

	col_condition, col_probe_type, col_acc = COL_CONDITION, COL_PROB_TYPE, COL_ACC
	fp_df = f'letterstring_online_experiment/human_subjects_UW_results.parquet'
	df = pd.read_parquet(fp_df)

	# simple plot
	df_acc = df.groupby([col_condition, col_probe_type])[col_acc].mean()
	df_sum = df.groupby([col_condition, col_probe_type])[col_acc].sum()
	df_sum.name = 'nr_success'
	df_count = df.groupby([col_condition, col_probe_type])[col_acc].count()
	df_count.name = 'nr_trials'
	#df_all = pd.concat([df_acc, df_sum, df_count], axis=1)
	new_dicts_d = {}
	prob_types = df_acc[conditions[0]].index.to_list()
	for cond in conditions:
		new_d = {}
		acc = df_acc[cond].values
		counts = df_count[cond].values
		nr_success = df_sum[cond].values
		ci_lower, ci_upper = proportion_confint(nr_success, counts)
		all_lower_err = acc - ci_lower
		all_upper_err = ci_upper - acc
		all_zerogen_err = np.array([all_lower_err, all_upper_err])
		new_d[COL_ALL_ERR] = all_zerogen_err
		new_d[COL_ALL_ACC] = acc
		new_dicts_d[cond] = new_d

	result_dicts = [v for k, v in new_dicts_d.items() if k in conditions]

	results_dir = RESULTS_DIR # 'human_vs_GPT3/'
	colors, legend_li = get_colors_legend(conditions)
	title = 'Human Subjects zero-generalization problems'.title()
	fp = results_dir + 'humans_zerogen_acc_comparison.png'
	do_plot(result_dicts, colors, legend_li, prob_types, fp, title)

	# original study
	hs_fp = 'behavioral_results/zerogen_acc.npz'
	results_dict = dict(np.load(hs_fp))
	#  adjust order to align with UW results
	trans_order = ['succ', 'pred', 'add_letter', 'remove_redundant', 'fix_alphabet', 'sort']  # acc. to gen problems
	adjust_order(results_dict, trans_order)
	conditions = ['original_study', 'original']
	result_dicts = [results_dict] + [v for k, v in new_dicts_d.items() if k in conditions]
	title = 'Human Subjects UCLA vs UW'.title()
	fp = results_dir + 'zerogen_acc_humans_ucla_uw.png'
	colors, legend_li = get_colors_legend(conditions)
	legend_li = ['UCLA', 'UW']
	do_plot(result_dicts, colors, legend_li, prob_types, fp, title)


	# comparison GPT-3 and humans
	versions = ['1_real', '3_synthetic']
	conditions_comparison = CONDITIONS[1:]
	colors = COLORS_TWO
	legend_list = ['GPT-3', 'Human']
	for vrs, cond in zip(versions, conditions_comparison):
		results_dicts, _, legend_li = get_results_versions([vrs])
		ttle = legend_li[0].title()
		result_dict = dict(results_dicts[0])
		adjust_order(result_dict, TRANS)
		title = f'GPT-3 vs. Human | {ttle}'
		t_file = legend_li[0].lower().replace(' ', '_')
		fp = results_dir + f'zerogen_acc_comparison_{t_file}.png'
		results_dicts = [result_dict, new_dicts_d[cond]]
		legend_li.append(cond)
		do_plot(results_dicts, colors, legend_list, prob_types, fp, title)


if __name__ == '__main__':
	plot_humansubjects_uw()
	plot_modified_gpt3()
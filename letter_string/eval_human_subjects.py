import json
import re

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from letter_string.generate_human_subjects_problems import CONDITIONS


COL_CONDITION, COL_PROB_TYPE = 'col_condition', 'col_probe_type'
COL_ACC = 'col_acc'


def get_all_problems():
    conditions = CONDITIONS
    all_probs = {}
    for k, cond in enumerate(conditions):
        suffix = f'_{cond}'
        all_prob = np.load(f'./all_prob{suffix}.npz', allow_pickle=True)['all_prob']
        all_prob_dict = all_prob.item()
        all_probs[cond] = all_prob_dict
    return all_probs


def evaluate_answers():
    def eval_response(row):
        resp = row[col_resp]
        key = list(resp.keys())[0]
        letters = resp[key] # .replace('[', '').replace(']', '').strip()
        letters = ''.join(re.findall(r'[a-zA-Z]+', letters))
        if 'original' in key:
            cond = 'original'
        elif 'synthetic' in key:
            cond = 'modified_synthetic'
        else:
            cond = 'modified'
        prob_type = key.replace(f'{cond}_', '').replace('_gen', '')  # e.g., {"modified_synthetic_remove_redundant_gen":"f"}
        prob_ind = int(row[col_probe])
        corr_answer = all_probs[cond][prob_type]['prob'][prob_ind][1][1]
        len_diff = len(set(letters).symmetric_difference(set(corr_answer)))
        corr_answer = ''.join(corr_answer)
        acc = 1 if corr_answer == letters else 0
        return cond, prob_type, letters, corr_answer, acc
        #return f'{cond}%%%{prob_type}%%%{letters}'

    col_resp, col_probe, col_users, col_date, col_runid = 'response', 'prob_ind', 'Test-person', 'timestamp', 'run_id'
    col_condition, col_probe_type, col_letters_answer, col_corr_answer = COL_CONDITION, COL_PROB_TYPE,\
                                                                             'letters_answer', 'corr_answer'
    col_acc = COL_ACC

    all_probs = get_all_problems()
    file_names = ['letterstring-exp', 'copy-of-letterstring-exp']
    dfs = []
    for i, fname in enumerate(file_names):
        fp_xl = f'letterstring_online_experiment/{fname}.csv'
        dfi = pd.read_csv(fp_xl, header=0)
        dfi[col_runid] = dfi[col_runid] + i*1000
        dfs.append(dfi)
    df = pd.concat(dfs)
    df[col_date] = pd.to_datetime(df[col_date])
    df_r = pd.DataFrame(df.groupby(col_runid)[col_date].max())
    df_r = df_r.join(pd.DataFrame(df.groupby(col_runid)[col_probe].count()), lsuffix='del')
    dt_cutoff = pd.to_datetime('2023-11-15 00:00:00')
    dt_cutoff_max = pd.to_datetime('2023-11-30 00:00:00')
    df_r = df_r[df_r[col_date] > dt_cutoff]
    #df_r = df_r[df_r[col_date] < dt_cutoff_max]
    nr_problems = 18
    df_r = df_r[df_r[col_probe] >= nr_problems] # total nr of problems
    val_runs = df_r.index
    df = df[df[col_runid].isin(val_runs)]
    nr_runs = len(val_runs)
    print('nr valid runs', nr_runs, 'nr credit codes', df['credit_code'].count())
    df = df[[col_probe, col_resp, col_runid]]
    df = df[~df[col_probe].isna()]
    assert df.shape[0] == nr_problems * nr_runs  # control
    df[col_resp] = df[col_resp].apply(json.loads)
    df[[col_condition, col_probe_type, col_letters_answer, col_corr_answer, col_acc]] = df.apply(eval_response, axis=1).to_list()
    print(df[col_acc].mean(), df.groupby(col_condition)[col_acc].mean())
    print(df.groupby([col_condition, col_probe_type])[col_acc].mean())
    fp_df = f'letterstring_online_experiment/human_subjects_UW_results.parquet'
    df.to_parquet(fp_df)


if __name__ == '__main__':
    evaluate_answers()
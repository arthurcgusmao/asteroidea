import os, sys
import pandas as pd
import numpy as np
import re


if not os.path.basename(os.getcwd()) == 'results':
    sys.exit("This script should be run from 'results/', please cd in there.")


def time_when_asteroidea_achieves_ll(df, ll):
    indexes = df.index[df['Log-Likelihood'] >= ll].tolist()
    if len(indexes) > 0:
        return df.iloc[min(indexes)]['Elapsed Time'], df.iloc[min(indexes)]['Log-Likelihood']
    else:
        return np.nan, np.nan


def get_asteroidea_best_ll(df):
    last_row_index = max(df.index.tolist())
    return df.iloc[last_row_index]['Elapsed Time'], df.iloc[last_row_index]['Log-Likelihood']


def get_dset_instance_number(results_fname, dset_name):
    pattern = r'(?<=___{}_).*?(?=\.)'.format(dset_name)
    return re.search(pattern, results_fname).group()


def compile_results_for_each_dset_instance():
    # get a list of all dirs
    dirs = [x for x in os.listdir('./') if os.path.isdir(x)]
    # for each instance of the datasets (dir_), compile the thing
    # dir_ is the instance folder name, for example: alarm_02
    for dir_ in dirs:
        # check that asteroidea and problog dirs exist
        if os.path.isdir(dir_ + '/asteroidea') and os.path.isdir(dir_ + '/problog'):
            # get a list of all experiments for each algorithm
            ast_exps = [x for x in os.listdir(dir_ + '/asteroidea') if os.path.isdir(dir_ + '/asteroidea/' + x)]
            prob_exps = [x for x in os.listdir(dir_ + '/problog') if os.path.isdir(dir_ + '/problog/' + x)]
            # get the lastest experiment (higher timestamp)
            ast_latest_exp = max(ast_exps)
            prob_latest_exp = max(prob_exps)

            # get problog dataframe
            prob_df = pd.read_csv('{}/problog/{}/problog___{}___{}.csv'.format(
                dir_, prob_latest_exp, dir_, prob_latest_exp))

            # get asteroidea dataframes (one for each dataset size and missing rate)
            ast_dir = '{}/asteroidea/{}'.format(dir_, ast_latest_exp)
            ast_files = os.listdir(ast_dir)
            # asteroidea files are in the form of asteroidea___dataset_name___dataset_XXXsize_XXXmissing.XX___timestamp
            # we are going to use regex to help us find the substring
            ast_dfs_dict = {}
            pattern = r'dataset_.*?missing\.(pl|csv)'
            for ast_file in ast_files:
                ast_dset_name = re.search(pattern, ast_file).group()
                ast_dfs_dict[ast_dset_name] = os.path.join(ast_dir, ast_file)

            # create an array to store the time for each algorithm
            results = []

            # for the respective filename in each problog's dataframe line:
            for index,row in prob_df.iterrows():
                prob_time = row['time']
                row_ll = row['log-likelihood']
                row_fname = row['filename']
                # revome problog prefix in case it is present
                row_fname = row_fname.replace('problog_', '')

                # get the respective asteroidea dataframe
                if row_fname in ast_dfs_dict:
                    if os.path.isfile(ast_dfs_dict[row_fname]):
                        ast_df = pd.read_csv(ast_dfs_dict[row_fname])
                        # find time in dataframe when asteroidea reached a better LL value
                        ast_time, ast_ll = time_when_asteroidea_achieves_ll(ast_df, row_ll)
                        best_ast_ll_time, best_ast_ll = get_asteroidea_best_ll(ast_df)
                    else:
                        ast_time = np.nan
                        ast_ll = np.nan
                        best_ast_ll_time = np.nan
                        best_ast_ll = np.nan
                else:
                    ast_time = np.nan
                    ast_ll = np.nan
                    best_ast_ll_time = np.nan
                    best_ast_ll = np.nan

                # insert both problog time and asteroidea time into a new array
                results.append({'filename': row_fname, 'asteroidea_time': ast_time, 'problog_time': prob_time,
                                'asteroidea_ll': ast_ll, 'problog_ll': row_ll,
                                'asteroidea_best_ll': best_ast_ll, 'asteroidea_best_ll_time': best_ast_ll_time})

            # finally, save results to a .csv file
            results_fname = os.path.join(dir_, 'problog_vs_asteroidea___{}.csv'.format(dir_))
            pd.DataFrame(results).to_csv(results_fname)


def compile_final_results(dataset_names):
    # get a list of all dirs
    dirs = [x for x in os.listdir('./') if os.path.isdir(x)]
    # for each instance of the datasets (dir_), compile the thing
    # dir_ is the instance folder name, for example: alarm_02
    for dset_name in dataset_names:
        dset_final_results = {}
        for dir_ in dirs:
            if dset_name in dir_:
                results_fname = os.path.join(dir_, 'problog_vs_asteroidea___{}.csv'.format(dir_))
                if os.path.isfile(results_fname):
                    dset_instance = get_dset_instance_number(results_fname, dset_name)
                    df = pd.read_csv(results_fname)
                    for index,row in df.iterrows():
                        row_fname = row['filename']
                        row_ast = row['asteroidea_time']
                        row_prob = row['problog_time']
                        row_ast_ll = row['asteroidea_ll']
                        row_prob_ll = row['problog_ll']
                        row_best_ast_ll = row['asteroidea_best_ll']
                        row_best_ast_ll_time = row['asteroidea_best_ll_time']
                        if not row_fname in dset_final_results:
                            dset_final_results[row_fname] = {}
                        dset_final_results[row_fname]['asteroidea_{}'.format(dset_instance)] = row_ast
                        dset_final_results[row_fname]['problog_{}'.format(dset_instance)] = row_prob
                        dset_final_results[row_fname]['filename'] = row_fname
        pd.DataFrame(dset_final_results).to_csv('{}_final_results.csv'.format(dset_name))

compile_results_for_each_dset_instance()
compile_final_results(['alarm', 'ship_energy_plant'])

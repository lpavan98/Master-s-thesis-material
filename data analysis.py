import pandas as pd
from scipy.stats import spearmanr
import numpy as np

data_path = 'C:/Users/loren/Downloads/concept-learning-experiment-draft (7).csv'
data = pd.read_csv(data_path)
with open('Baseline Model Results.txt', 'r') as file:
    lines = file.readlines()

def label_pragmatic_in_control_group():
    control_group_run_ids = listener_lot_data['run_id'].unique()
    # Create a DataFrame for each participant
    participant_dataframes = [listener_lot_data[listener_lot_data['run_id'] == run_id].copy() for run_id in control_group_run_ids]

    # On the right, the indexes of the lines corresponding to pragmatic trials in the control group
    label_modifications = {
        (1, 'P'): [3, 9],
        (2, 'P'): [2, 3],
        (3, 'P'): [1, 8, 10],
        (4, 'P'): [1],
        (1, 'Q'): [2, 6],
        (2, 'Q'): [7, 9],
        (3, 'Q'): [4],
        (4, 'Q'): [3, 6, 9],
    }

    for participant_df in participant_dataframes:
        grouped = participant_df.groupby(['block_number', 'word_used'])
        for (block_number, word_used), indices in label_modifications.items():
            if (block_number, word_used) in grouped.groups:
                group = grouped.get_group((block_number, word_used))
                for index_offset in indices:
                    if index_offset < len(group):
                        absolute_index = group.index[index_offset]
                        participant_df.loc[absolute_index, 'label'] = 'p'

    # Return a single dataframe
    return pd.concat(participant_dataframes, ignore_index=True)

def m_or_q_experimental_group():
    # Create a DataFrame for each participant
    experimental_group_run_ids = listener_data['run_id'].unique()
    listener_data['m_or_q'] = ''  # Initialize the new label column
    participant_dataframes = [listener_data[listener_data['run_id'] == run_id].copy() for run_id in experimental_group_run_ids]

    # Rows to label for M-implicatures and Q-implicatures
    m_label_modifications = {
        1: [4, 8, 14],
        2: [5, 10, 14, 19],
        3: [4, 8],
        4: [4, 8, 12],
    }
    q_label_modifications = {
        1: [19],
        2: [],
        3: [16, 20],
        4: [19],
    }

    # Apply label modifications to each participant's DataFrame
    for participant_df in participant_dataframes:
        grouped = participant_df.groupby('block_number')

        # Assign 'm' labels
        for block_number, indices in m_label_modifications.items():
            if block_number in grouped.groups:
                group = grouped.get_group(block_number)
                for index_offset in indices:
                    if index_offset < len(group):
                        absolute_index = group.index[index_offset]
                        participant_df.loc[absolute_index, 'm_or_q'] = 'm'

        # Assign 'q' labels
        for block_number, indices in q_label_modifications.items():
            if block_number in grouped.groups:
                group = grouped.get_group(block_number)
                for index_offset in indices:
                    if index_offset < len(group):
                        absolute_index = group.index[index_offset]
                        participant_df.loc[absolute_index, 'm_or_q'] = 'q'

    return pd.concat(participant_dataframes, ignore_index=True)

# Label would-be conversational implicatures
def m_or_q_control_group():
    # Look at trials corresponding to pragmatic trials
    p_rows = listener_lot_data[listener_lot_data['label'] == 'p']

    run_ids = listener_lot_data['run_id'].unique()
    # Loop through each participant
    for run_id in run_ids:
        participant_data = p_rows[p_rows['run_id'] == run_id]

        for block_number in participant_data['block_number'].unique():
            block_rows = participant_data[participant_data['block_number'] == block_number]

            if block_number == 2:
                listener_lot_data.loc[block_rows.index, 'm_or_q'] = 'm'
            elif block_number == 3:
                listener_lot_data.loc[block_rows.index[:2], 'm_or_q'] = 'm'
                listener_lot_data.loc[block_rows.index[2:], 'm_or_q'] = 'q'
            else:
                listener_lot_data.loc[block_rows.index[:3], 'm_or_q'] = 'm'
                listener_lot_data.loc[block_rows.index[3:], 'm_or_q'] = 'q'

    return listener_lot_data

# Exclude first 2 trials in blocks 1, 3, 4 and exclude 1st and 3rd trial in block 2, as they are the first trials in which messages appear
def exclude_rows(df):
    if df['block_number'].iloc[0] in [1, 3, 4]:
        return df.iloc[2:]
    else:
        return df.drop(df.index[[0, 2]])

# Get proportion of control trials done right and wrong for both types of control trials for each participant
def get_control_trials_proportion(exp_data):
    # Count occurrences of each label for each participant
    label_counts_per_participant = exp_data.groupby(['run_id', 'label']).size().unstack(fill_value=0)
    # Calculate proportions of control trials
    control_proportions = label_counts_per_participant.copy()
    control_proportions['cr_proportion'] = control_proportions['cr'] / (control_proportions['cr'] + control_proportions['cw'])
    control_proportions['cer_proportion'] = control_proportions['cer'] / (control_proportions['cer'] + control_proportions['cew'])
    control_proportions['cr_cer_proportion'] = (control_proportions['cr'] + control_proportions['cer']) / (control_proportions['cr'] + control_proportions['cer'] + control_proportions['cw'] + control_proportions['cew'])
    return control_proportions

# Filter out runs with less than 70% control trials passed
def filter_based_on_controls(correctness_proportions, exp_data):
    correctness_proportions = correctness_proportions[(correctness_proportions['cr_cer_proportion'] >= 0.7)]
    return exp_data[exp_data['run_id'].isin(correctness_proportions.index)]

# Calculate average and per-participant accuracy
def calculate_accuracy(exp_data):
    average_accuracy = exp_data['correct'].mean()
    per_participant_accuracy = exp_data.groupby('run_id')['correct'].mean()

    return average_accuracy, per_participant_accuracy

# Get probability of answering correctly each trial
def results_to_probabilities(pr, control_true_false):
    # In the control group, sort the data: first come the lines where the message is P, then those where the message is Q
    if control_true_false:
        pr = pd.concat([pr[pr['word_used'] == 'P'], pr[pr['word_used'] == 'Q']], ignore_index=True)
    # Calculate the probability of giving the correct response to each trial, after giving an id to each trial
    pr['trial_id'] = pr.groupby(['run_id', 'block_number']).cumcount() + 1
    pr['trial_id'] = pr['block_number'].astype(str) + "-" + pr['trial_id'].astype(str)
    pr = pr.groupby('trial_id')['correct'].mean()
    # Back to dataframe, in the previous line it became a series
    pr = pr.reset_index()
    # Reorder the dataframe
    pr[['block', 'position']] = pr['trial_id'].str.split('-', expand=True)
    pr[['block', 'position']] = pr[['block', 'position']].astype(float)
    pr = pr.sort_values(by=['block', 'position'])

    return pr

# Preprocessing
# Participant with run_id 30 did the experiment twice. Exclude the second run
data = data[data['run_id'] != 30]
# Get listener data for experimental group.
listener_data = data[data['task'] == 'listener']
# Get listener data for control group
listener_lot_data = data[data['task'] == 'listener_LOT_only']
# Get speaker data
speaker_data = data[data['task'] == 'speaker']
# Get proportion of control trials done right and wrong for both types of control trials
control_proportions_per_participant_experimental = get_control_trials_proportion(listener_data)
control_proportions_per_participant_lot_only = get_control_trials_proportion(listener_lot_data)
# Filter out runs with accuracies of controls below 70%
listener_data = filter_based_on_controls(control_proportions_per_participant_experimental, listener_data)
listener_lot_data = filter_based_on_controls(control_proportions_per_participant_lot_only, listener_lot_data)
# Find missing labels
listener_lot_data = label_pragmatic_in_control_group()
listener_lot_data = m_or_q_control_group()
listener_data = m_or_q_experimental_group()

# Get probabilities of obtaining correct response, for each trial
listener_lot_data['correct'] = listener_lot_data['correct'].astype(int)
listener_data['correct'] = listener_data['correct'].astype(int)
probabilistic_results = results_to_probabilities(listener_data, False)
probabilistic_results_control = results_to_probabilities(listener_lot_data, True)
probabilistic_results.to_csv("experiment_probres.csv")
probabilistic_results_control.to_csv("experiment_probrescontrol.csv")

# Remove trials corresponding to the first occurrences of each message
listener_data = listener_data.groupby(['run_id', 'block_number'], as_index=False).apply(exclude_rows).reset_index(drop=True)
listener_lot_data = (listener_lot_data.groupby(['run_id', 'block_number', 'word_used'])).apply(lambda group: group.iloc[1:]).reset_index(drop=True)

m_data = listener_data[listener_data['m_or_q'] == 'm']
q_data = listener_data[listener_data['m_or_q'] == 'q']
# Get trials in the control group that correspond to the pragmatic trials in the experimental condition
would_be_pragmatic_trials = listener_lot_data[listener_lot_data['label'] == 'p']
# Get pragmatic trials
pragmatic_trials = listener_data[listener_data['label'].isin(['pr', 'phr', '-'])]
# Get control trials
controls_experimental = listener_data[listener_data['label'].isin(['cr', 'cw', 'cer', 'cew'])]
controls_control = listener_lot_data[listener_lot_data['label'].isin(['cr', 'cw', 'cer', 'cew'])]
# Neither control trials nor pragmatic trials
lot_only_no_control_or_pragmatics = listener_lot_data[~listener_lot_data['label'].isin(['cr', 'cw', 'cer', 'cew', 'p'])]
experimental_no_control_or_pragmatics = listener_data[~listener_data['label'].isin(['cr', 'cw', 'cer', 'cew', 'pr', 'phr', '-'])]

# Count occurrences of each label for each participant
label_counts_per_participant = listener_data.groupby(['run_id', 'label']).size().unstack(fill_value=0)

# Look at the Results of the Baseline Model, comparing the probabilities of responding correctly to trials to the most probable responses of the model
# Count when pr >= p1, p2, and p3
pr_is_higher_or_equal = 0  
# Count when pr is lower than p1, p2, and p3
pr_is_lower = 0  

for line in lines:
    if "Probability to respond correctly:" in line and 'pragmatic trial' not in line:
        # Extract the probabilities from the line
        parts = line.split(',')
        p1 = float(parts[0].split('=')[1].strip())
        p2 = float(parts[1].split('=')[1].strip())
        p3 = float(parts[2].split('=')[1].strip().split('.')[0])
        pr = float(parts[-1].split(':')[1].strip())

        # Check if pr is higher than p1, p2, and p3
        if pr >= p1 and pr >= p2 and pr >= p3:
            pr_is_higher_or_equal += 1
        else:
            #print(line)
            #print(max(p1,p2,p3) - pr)
            pr_is_lower += 1

#print(f"Number of times pr >= p1, p2, p3: {pr_is_higher_or_equal}")
#print(f"Number of times pr < p1, p2, p3: {pr_is_lower}")

# Calculate accuracies
average_accuracy_lot = lot_only_no_control_or_pragmatics['correct'].mean()
average_accuracy_exp = experimental_no_control_or_pragmatics['correct'].mean()
# Calculate accuracy per block
accuracy_per_block_lot = lot_only_no_control_or_pragmatics.groupby('block_number')['correct'].mean()
accuracy_per_block_exp = experimental_no_control_or_pragmatics.groupby('block_number')['correct'].mean()

print("Lot Only (No Control or Pragmatics):")
print("Average Accuracy:", average_accuracy_lot)
print("Accuracy per Block:")
print(accuracy_per_block_lot)
print("\nExperimental (No Control or Pragmatics):")
print("Average Accuracy:", average_accuracy_exp)
print("Accuracy per Block:")
print(accuracy_per_block_exp)

print(calculate_accuracy(listener_lot_data))
print(calculate_accuracy(listener_data))

""" # Prepare data for the generalized linear mixed-effects models
# The code for running the generalized linear mixed-effects models is reported in the results section of the thesis. It is R code. To run it, you only need to first include the lines "library(lme4)", "library(Matrix)" and read the csv files produced below. Group_binary should always be used as the name for the fixed variable.
# Control vs experimental
listener_lot_data['Group'] = 'Control'
listener_data['Group'] = 'Experimental'
combined_data = pd.concat([listener_data, listener_lot_data], ignore_index=True)
combined_data['Group_binary'] = combined_data['Group'].apply(lambda x: 1 if x == 'Experimental' else 0)
combined_data.to_csv("mm1.csv")

# Pragmatic trials vs would be pragmatic trials
would_be_pragmatic_trials['Group'] = 'Would_Be_Pragmatic'
pragmatic_trials['Group'] = 'Pragmatic_Trials'
combined_data = pd.concat([would_be_pragmatic_trials, pragmatic_trials], ignore_index=True)
combined_data['Group_binary'] = combined_data['Group'].apply(lambda x: 1 if x == 'Pragmatic_Trials' else 0)
combined_data.to_csv("mm2.csv")

# Pragmatic trials vs normal (without controls) trials
experimental_no_control_or_pragmatics['Group'] = 'Regular_trials'
pragmatic_trials['Group'] = 'Pragmatic_Trials'
combined_data = pd.concat([experimental_no_control_or_pragmatics, pragmatic_trials], ignore_index=True)
combined_data['Group_binary'] = combined_data['Group'].apply(lambda x: 1 if x == 'Regular_trials' else 0)
combined_data.to_csv("mm3.csv")

# M vs Q
m_data['Group'] = 'm'
q_data['Group'] = 'q'
combined_data = pd.concat([m_data, q_data], ignore_index=True)
combined_data['Group_binary'] = combined_data['Group'].apply(lambda x: 1 if x == 'm' else 0)
combined_data.to_csv("mm4.csv")

# Normal trials vs control trials without exclusion of participants
# Remember to comment out the two lines using the function filter_based_on_controls
normal_trials = pd.concat([experimental_no_control_or_pragmatics, lot_only_no_control_or_pragmatics], ignore_index=True)
normal_trials['Group'] = 'Regular_trials'
control_trials = pd.concat([controls_experimental, controls_control], ignore_index=True)
control_trials['Group'] = 'Control trials'
combined_data = pd.concat([normal_trials, control_trials], ignore_index=True)
combined_data['Group_binary'] = combined_data['Group'].apply(lambda x: 1 if x == 'Regular_trials' else 0)
combined_data.to_csv("mm5noexclusion.csv") """


file_names = ['pr0.50.csv', 'pr0.51.csv', 'pr0.52.csv', 'pr0.525.csv', 'pr0.53.csv', 'pr0.54.csv', 
              'pr0.55.csv', 'pr0.575.csv', 'pr0.60.csv', 'pr0.625.csv', 
              'pr0.65.csv', 'pr0.70.csv', 'pr0.75.csv', 'pr0.80.csv', 
              'pr0.85.csv', 'pr0.90.csv', 'a1c0.csv', 'a2c0.csv', 'a3c0.csv', 
              'a4c0.csv', 'a5c0.csv']

# Read the experimental results
exp_results = pd.read_csv('experiment_probres.csv')
exp_results = exp_results.groupby('block')['correct'].apply(list).to_dict()
exp_control_results = pd.read_csv("experiment_probrescontrol.csv")
exp_control_results = exp_control_results.groupby("block")["correct"].apply(list).to_dict()

control_model_results = pd.read_csv("probability of correct responses control group Baseline Model.csv")

# Deal with control group first
observed_flat_control = np.concatenate([np.array(v) for v in exp_control_results.values()])
generated_flat_control = np.concatenate([control_model_results[col].to_numpy() for col in control_model_results.columns])

spearman_correlation_control, p_value_control = spearmanr(observed_flat_control, generated_flat_control)
total_absolute_differences_control = sum(np.abs(observed_flat_control - generated_flat_control))

print(f"\nResults for Control Group Model vs Experimental Control:")
print(f"Spearman Correlation: {spearman_correlation_control:.4f} (p-value: {p_value_control:.4f})")
print(f"Total Absolute Differences: {total_absolute_differences_control:.4f}")

# Deal with experimental group
observed_flat = np.concatenate([np.array(v) for v in exp_results.values()])

for file_name in file_names:
    df = pd.read_csv(file_name)

    generated_flat = np.concatenate([df[col].to_numpy() for col in df.columns])

    spearman_correlation, p_value = spearmanr(observed_flat, generated_flat)
    total_absolute_differences = sum(np.abs(observed_flat - generated_flat))

    print(f"\nResults for {file_name}:")
    print(f"Spearman Correlation: {spearman_correlation:.4f} (p-value: {p_value:.4f})")
    print(f"Total Absolute Differences: {total_absolute_differences:.4f}")
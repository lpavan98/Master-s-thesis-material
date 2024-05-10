import itertools
from copy import deepcopy
import numpy as np
import pandas as pd

likelihood_modifier = 0.9

# Experimental results
exp_results = pd.read_csv('experiment_probres.csv')
exp_results = exp_results.groupby('block')['correct'].apply(list).to_dict()
exp_results_control = pd.read_csv('experiment_probrescontrol.csv')
exp_results_control = exp_results_control.groupby('block')['correct'].apply(list).to_dict()

# Unallowed hypotheses with "and" see a subfeature bonded by "and" to another subfeature of the same feature, as for example "blue and red".
# Unallowed hypotheses with "or" see a subfeature bonded by "or" to its negation, as for example "blue or not blue".
def generate_unallowed_hypotheses():
    # Divide literals in sets based on features (Sizes, Colors, sHapes).
    # Literals with the "n" in front are negations of the literal that follows "n".
    sets_of_literals = [
        ['s1', 's2', 's3', 'ns1', 'ns2', 'ns3'],
        ['c1', 'c2', 'c3', 'nc1', 'nc2', 'nc3'],
        ['h1', 'h2', 'h3', 'nh1', 'nh2', 'nh3']
    ]
    unallowed_hypotheses = []
    # Create a string for each hypothesis with "and"
    for literals in sets_of_literals:
        for comb in list(itertools.combinations(literals, 2)):
            hypothesis = ' '.join(f'{x} {c}' for x, c in zip(comb, ('and',))) + f' {comb[-1]}'
            unallowed_hypotheses.append(hypothesis)
    # Deal with hypotheses with "or"
    unallowed_hypotheses.extend(['s1 or ns1', 's2 or ns2', 's3 or ns3', 'c1 or nc1', 'c2 or nc2', 'c3 or nc3', 'h1 or nh1', 'h2 or nh2', 'h3 or nh3'])

    return unallowed_hypotheses

def check_elements_not_in_string(elements, string_to_check):
    for element in elements:
        if element in string_to_check:
            return False
    return True   

# Transform a list of costs in a list of probabilities
def cost_to_probability(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def create_hypotheses_space():
    cost_literals = -1
    cost_booleans = -1
    elements = {"literals": ['s1', 's2', 's3', 'c1', 'c2', 'c3', 'h1', 'h2', 'h3', 
                            'ns1', 'ns2', 'ns3', 'nc1', 'nc2', 'nc3', 'nh1', 'nh2', 'nh3'],
                "costs": [cost_literals] *9 + [cost_literals + cost_booleans] * 9}
    conjunctions = ['and', 'or']
    # p_hypotheses will store the hypotheses, as long as their costs and probabilities, for the concept represented by "p".
    p_hypotheses = {"hypotheses": [],"costs": [],"probabilities": []}

    # Find all hypotheses with 1 literal and their costs
    for i in zip(elements["literals"], elements["costs"]):
        p_hypotheses["hypotheses"].append(i[0])
        p_hypotheses['costs'].append(i[1])

    # Find all hypotheses with 2 literals and their costs
    for comb in list(itertools.combinations(elements["literals"], 2)):
        for conj in itertools.product(conjunctions, repeat=len(comb) - 1):
            hypothesis = ' '.join(f'{x} {c}' for x, c in zip(comb, conj)) + f' {comb[-1]}'
            cost_for_literals = sum(elements["costs"][elements["literals"].index(literal)] for literal in comb)
            # Store hypotheses that are not in the list of unallowed hypotheses
            if check_elements_not_in_string(generate_unallowed_hypotheses(), hypothesis):
                p_hypotheses["hypotheses"].extend([hypothesis, 'n(' + hypothesis + ')'])
                p_hypotheses["costs"].extend([cost_for_literals + cost_booleans, cost_for_literals + cost_booleans * 2])

    # Calculate the probability of hypotheses based on their cost
    p_hypotheses['probabilities'] = cost_to_probability(p_hypotheses['costs'])
    # Costs are no longer needed
    del p_hypotheses["costs"]
    # At the onset, the hypotheses for the two concepts have the same probabilities
    q_hypotheses = deepcopy(p_hypotheses)

    return p_hypotheses, q_hypotheses

# Create lexica containing hypotheses for both p and q, as well as probabilities for the combinations of such hypotheses
def create_lexica(p_hypotheses, q_hypotheses):
    lexica = {"p": [],"q": [],"probabilities": []}
    for i in range(len(p_hypotheses["hypotheses"])):
        for j in range(len(q_hypotheses["hypotheses"])):
            lexica["p"].append(p_hypotheses['hypotheses'][i])
            lexica["q"].append(q_hypotheses['hypotheses'][j])
            lexica["probabilities"].append(p_hypotheses["probabilities"][i] * q_hypotheses["probabilities"][j])
    return lexica

def check_condition(condition, item):
    # Check if the whole condition is a negation
    negate_eventually = False
    if condition[-1] == ')':
        condition = condition[2:-1]
        negate_eventually = True
    
    # Split the condition into literals and conjunctions
    literals = [literal for literal in condition.split() if literal not in ['and', 'or']]
    conjunction = [lit for lit in condition.split() if lit in ['and', 'or']]
    # Initialize flags to track the presence of literals in the item
    literal_present = {literal: False for literal in literals if literal != 'and' and literal != 'or'}

    for literal in literals:
        # Check negations of literals
        if literal[0] == 'n' and literal[1:] not in item:
                literal_present[literal] = True
        # Check normal literals
        elif literal in item:
            literal_present[literal] = True
    if not negate_eventually:
        if 'and' in conjunction:
            return all(literal_present.values())
        else:
            return any(literal_present.values())
    else:
        if 'and' in conjunction:
            return not all(literal_present.values())
        else:
            return not any(literal_present.values())
        
def update_beliefs_based_on_feedback(message, feedback, cuncurrent_reasoning, likelihood_modifier_concurrent):
    if message == 'p':
        x_hypotheses = p_hypotheses
        y_hypotheses = q_hypotheses
    else:
        x_hypotheses = q_hypotheses
        y_hypotheses = p_hypotheses

    # Iterate through each hypothesis and its index
    for i, hypothesis in enumerate(x_hypotheses["hypotheses"]): 
        # Check if the hypothesis describes the feedback item, then update probability of hypotheses
        if check_condition(hypothesis, feedback):
            x_hypotheses["probabilities"][i] *= likelihood_modifier
            if cuncurrent_reasoning:
                y_hypotheses["probabilities"][i] *= (1 - likelihood_modifier_concurrent)
        else:
            x_hypotheses["probabilities"][i] *= (1 - likelihood_modifier)
            if cuncurrent_reasoning:
                y_hypotheses["probabilities"][i] *= likelihood_modifier_concurrent

    # Normalize probabilities
    total_probability = sum(x_hypotheses["probabilities"])
    x_hypotheses['probabilities'] = [prob / total_probability for prob in x_hypotheses["probabilities"]]
    if cuncurrent_reasoning:
        total_probability = sum(y_hypotheses["probabilities"])
        y_hypotheses['probabilities'] = [prob / total_probability for prob in y_hypotheses["probabilities"]]

    # Update lexica by recreating the relative dictionary
    lexica = create_lexica(p_hypotheses, q_hypotheses)

    return lexica

def generate_all_items():
    sets_of_literals = [
        ['s1', 's2', 's3'],
        ['c1', 'c2', 'c3'],
        ['h1', 'h2', 'h3']
    ]
    # Return all combinations of strings
    return [''.join(comb) for comb in itertools.product(*sets_of_literals)]

# Calculate probability to pick each object in a trial
def pick_object(message, trial, previous_word, L2, alpha, cost, cost_matters = False):
    all_items = generate_all_items()
    nl_list = []
    l_one_list = []
    p_obj1 = p_obj2 = p_obj3 = 0
    
    # Calculate the coefficient of proportionality
    for item in all_items:
        # List all hypotheses for message 'message' that are true for the item 'item'. Hypotheses for p are the same as those for q: they only differ in their probabilities. Since the probabilities do not matter here, the arbitrary decision to use hypotheses for p was taken.
        true_hypotheses = []
        for hypothesis in p_hypotheses["hypotheses"]: 
            if check_condition(hypothesis, item):
                true_hypotheses.append(hypothesis)

        nl = 0
        # Iterate over all lexica
        if cost_matters and previous_word != message:
            for i in range(len(lexica[message])):
                if lexica[message][i] in true_hypotheses:
                    nl += lexica['probabilities'][i] / (np.exp(cost) ** alpha)
                else:
                    nl += 10 ** (-9) / (np.exp(cost) ** alpha)
        else:
            for i in range(len(lexica[message])):
                if lexica[message][i] in true_hypotheses:
                    nl += lexica['probabilities'][i]
                else:
                    nl += 10 ** (-9)

        # Non-normalized probabilities of picking the objects displayed in the trial
        if item == trial[0]:
            p_obj1 = nl
        elif item == trial[1]:
            p_obj2 = nl
        elif item == trial[2]:
            p_obj3 = nl
        if L2:
            l_one_list.append(nl)

        # True hypotheses will be recreated for the next item at the next iteration of the loop.
        del true_hypotheses
        nl_list.append(nl)
        
    # k * nl_list[0] + k * nl_list[1] + ... = 1. Therefore the coefficient of proportionality k is
    k = 1 / sum(nl_list)
    
    if not L2:
        # Update the probability of selecting the items in the trial by using k
        p_obj1 *= k
        p_obj2 *= k
        p_obj3 *= k
        # The probabilities must sum up to 1
        total_probability = p_obj1 + p_obj2 + p_obj3
        if total_probability != 0:
            p_obj1 /= total_probability
            p_obj2 /= total_probability
            p_obj3 /= total_probability
        else:
            p_obj1 = p_obj2 = p_obj3 = 1 / 3

        return p_obj1, p_obj2, p_obj3
    
    else:
        # Keep track of the displayed items' probabilities
        p_obj1 = l_one_list.index(p_obj1)
        p_obj2 = l_one_list.index(p_obj2)
        p_obj3 = l_one_list.index(p_obj3)

        # Update the probability of selecting the items in the trial by using k
        l_one_list = list(map(lambda x: x * k, l_one_list))
        # Normalize probabilities
        totprob = sum(l_one_list)
        l_one_list = [prob / totprob for prob in l_one_list]

        nl = 0
        del nl_list
        nl_list = []
        # Calculate L2's normalization coefficient
        for listener_one in l_one_list:
            nl += (listener_one ** alpha) / np.exp(alpha * cost)
            nl_list.append(nl)
        k = 1 / sum(nl_list)

        p_obj1 = nl_list[p_obj1] * k
        p_obj2 = nl_list[p_obj2] * k
        p_obj3 = nl_list[p_obj3] * k
        total_probability = p_obj1 + p_obj2 + p_obj3
        if total_probability != 0:
            p_obj1 /= total_probability
            p_obj2 /= total_probability
            p_obj3 /= total_probability
        else:
            p_obj1 = p_obj2 = p_obj3 = 1 / 3
        
        return p_obj1, p_obj2, p_obj3

# Change stimuli into a more readable form, consistent with the rest of the program: for example, from '123' to 's1c2h3'.
def transform_stimuli(array):
    sets_of_literals = [
        ['s1', 's2', 's3'],
        ['c1', 'c2', 'c3'],
        ['h1', 'h2', 'h3']
    ]
    transformed_array = []
    for digit_group in array:
        transformed_group = ''.join([sets_of_literals[i][int(digit) - 1] for i, digit in enumerate(digit_group)])
        transformed_array.append(transformed_group)
    
    return transformed_array

def store_end_hypotheses(block):
    # Store and print best lexica   
    # Sort lexica based on probabilities
    sorted_lexica = sorted(zip(lexica["p"], lexica["q"], lexica["probabilities"]), key=lambda x: x[2], reverse=True)
    # Save the 5 most probable lexica
    for idx, (p, q, prob) in enumerate(sorted_lexica[:5]):
        list_lexica.append({
            'Block': block,
            'Index': idx + 1,
            'p': p,
            'q': q,
            'Probability': prob
        })
    # Print them
    for i in range(min(5, len(sorted_lexica))):
        p, q, prob = sorted_lexica[i]
        print(f"Lexica {i+1}: p={p}, q={q}, Probability={prob}")

    # Store and print best p_hypotheses
    # Sort p_hypotheses based on probabilities
    sorted_p_hypotheses_with_probabilities = sorted(zip(p_hypotheses["hypotheses"], p_hypotheses["probabilities"]), key=lambda x: x[1], reverse=True)
    # Store the top 5 most probable p_hypotheses with their probabilities
    for idx, (hypothesis, probability) in enumerate(sorted_p_hypotheses_with_probabilities[:5]):
        list_p_hypotheses.append({
            'Block': block,
            'Index': idx + 1,
            'Hypothesis': hypothesis,
            'Probability': probability
        })
    # Print them
    print("Top 5 most probable p_hypotheses with probabilities:")
    for i, (hypothesis, probability) in enumerate(sorted_p_hypotheses_with_probabilities[:5], start=1):
        print(f"{i}. Hypothesis: {hypothesis}, Probability: {probability}")

    # Store and print best q_hypotheses
    # Sort q_hypotheses based on probabilities
    sorted_q_hypotheses_with_probabilities = sorted(zip(q_hypotheses["hypotheses"], q_hypotheses["probabilities"]), key=lambda x: x[1], reverse=True)
    # Store the top 5 most probable p_hypotheses with their probabilities
    for idx, (hypothesis, probability) in enumerate(sorted_q_hypotheses_with_probabilities[:5]):
        list_q_hypotheses.append({
            'Block': block,
            'Index': idx + 1,
            'Hypothesis': hypothesis,
            'Probability': probability
        })
    # Print them
    print("Top 5 most probable q_hypotheses with probabilities:")
    for i, (hypothesis, probability) in enumerate(sorted_q_hypotheses_with_probabilities[:5], start=1):
        print(f"{i}. Hypothesis: {hypothesis}, Probability: {probability}")


def run_experiment(alpha, cost, likelihood_modifier_concurrent):

    global lexica
    # Probabilities to respond correctly will be stored here
    prob_correct = {i: [] for i in range(1, 5)}

    for i, stimulus_listener_x in enumerate(stimuli_listener):
        block = i + 1
        last_word = stimulus_listener_x[0]['word']
        for trial in stimulus_listener_x:
            stimuli = transform_stimuli(trial['stim'])
            # In pragmatic trials, use L2
            if 'pragmatics_wrong' in trial or 'pragmaticswrongh' in trial:
                p1, p2, p3 = pick_object(message=trial['word'], trial=stimuli, previous_word=last_word, L2=False, alpha=alpha, cost=cost)
            else:
                p1, p2, p3 = pick_object(message=trial['word'], trial=stimuli, previous_word=last_word, L2=False, alpha=alpha, cost=cost)
            last_word = trial['word']
            lexica = update_beliefs_based_on_feedback(trial['word'], stimuli[trial['correct']], cuncurrent_reasoning=False, likelihood_modifier_concurrent=likelihood_modifier_concurrent)
            # Probability to be correct is found
            pr = p1 if trial['correct'] == 0 else p2 if trial['correct'] == 1 else p3
            prob_correct[block].append(pr)
            print(f"Block {block}: p1={p1}, p2={p2}, p3={p3}. Probability to respond correctly: {pr}")

        store_end_hypotheses(block)

    return prob_correct

def run_experiment_control():

    global lexica
    # Probabilities to respond correctly will be stored here
    prob_correct = {i: [] for i in range(1, 5)}

    for i, stimulus_listener_x in enumerate(stimuli_listener):
        block = i + 1
        # Separate trials based on the message sent
        trials_p = [trial for trial in stimulus_listener_x if trial['word'] == 'p']
        trials_q = [trial for trial in stimulus_listener_x if trial['word'] == 'q']
        # Reunite trials: separation is achieved
        ordered_trials = trials_p + trials_q
        last_word = ordered_trials[0]['word']
        for trial in ordered_trials:
            stimuli = transform_stimuli(trial['stim'])
            p1, p2, p3 = pick_object(message=trial['word'], trial=stimuli, previous_word=last_word, L2=False, alpha=1, cost=0)
            last_word = trial['word']
            lexica = update_beliefs_based_on_feedback(trial['word'], stimuli[trial['correct']], cuncurrent_reasoning=False, likelihood_modifier_concurrent=0.5)
            # Probability to be correct is found
            pr = p1 if trial['correct'] == 0 else p2 if trial['correct'] == 1 else p3
            prob_correct[block].append(pr)
            print(f"Block {block}: p1={p1}, p2={p2}, p3={p3}. Probability to respond correctly: {pr}")

        store_end_hypotheses(block)

    return prob_correct

stimulus_listener_1 = [
    {'stim': ['213', '131', '322'], 'word': 'p', 'correct': 0},
    {'stim': ['121', '232', '323'], 'word': 'q', 'correct': 0},
    {'stim': ['122', '232', '233'], 'word': 'q', 'correct': 0},
    {'stim': ['223', '133', '332'], 'word': 'p', 'correct': 0},
    {'stim': ['133', '223', '332'], 'word': 'q', 'correct': 0, 'pragmatics_wrong': '223'},
    {'stim': ['223', '311', '312'], 'word': 'p', 'correct': 0},
    {'stim': ['133', '232', '323'], 'word': 'q', 'correct': 0},
    {'stim': ['123', '222', '323'], 'word': 'q', 'correct': 0},
    {'stim': ['123', '222', '323'], 'word': 'p', 'correct': 2, 'pragmatics_wrong': '123'},
    {'stim': ['112', '231', '322'], 'word': 'q', 'correct': 0},
    {'stim': ['233', '322', '311'], 'word': 'p', 'correct': 0},
    {'stim': ['323', '112', '232'], 'word': 'p', 'correct': 0},
    {'stim': ['333', '312', '321'], 'word': 'p', 'correct': 0},
    {'stim': ['333', '322', '311'], 'word': 'p', 'correct': 0},
    {'stim': ['333', '322', '311'], 'word': 'q', 'correct': 2, 'pragmatics_wrong': '333'},
    {'stim': ['213', '131', '112'], 'word': 'p', 'correct': 0},
    {'stim': ['212', '232', '222'], 'word': 'q', 'correct': 0},
    {'stim': ['313', '333', '321'], 'word': 'q', 'correct': 0},
    {'stim': ['213', '333', '221'], 'word': 'q', 'correct': 0},
    {'stim': ['213', '333', '221'], 'word': 'p', 'correct': 1, 'pragmaticswrongh': '213'},
    {'stim': ['211', '231', '322'], 'word': 'q', 'correct': 0},
    {'stim': ['133', '131', '323'], 'word': 'p', 'correct': 2},
]
stimulus_listener_2 = [
    {'stim': ['332', '221', '111'], 'word': 'p', 'correct': 1},
    {'stim': ['331', '221', '113'], 'word': 'p', 'correct': 1},
    {'stim': ['331', '312', '223'], 'word': 'q', 'correct': 1},
    {'stim': ['133', '212', '233'], 'word': 'q', 'correct': 1},
    {'stim': ['133', '212', '323'], 'word': 'q', 'correct': 1},
    {'stim': ['133', '212', '323'], 'word': 'p', 'correct': 2, 'pragmatics_wrong': '212'},
    {'stim': ['112', '323', '123'], 'word': 'q', 'correct': 0},
    {'stim': ['131', '222', '323'], 'word': 'q', 'correct': 1},
    {'stim': ['131', '122', '322'], 'word': 'q', 'correct': 1},
    {'stim': ['131', '222', '322'], 'word': 'q', 'correct': 1},
    {'stim': ['131', '222', '322'], 'word': 'p', 'correct': 2, 'pragmatics_wrong': '222'},
    {'stim': ['113', '211', '122'], 'word': 'p', 'correct': 2},
    {'stim': ['123', '112', '313'], 'word': 'p', 'correct': 0},
    {'stim': ['131', '211', '322'], 'word': 'p', 'correct': 2},
    {'stim': ['131', '211', '322'], 'word': 'q', 'correct': 1, 'pragmatics_wrong': '322'},
    {'stim': ['122', '123', '131'], 'word': 'q', 'correct': 0},
    {'stim': ['132', '223', '321'], 'word': 'p', 'correct': 1},
    {'stim': ['231', '321', '311'], 'word': 'p', 'correct': 1},
    {'stim': ['132', '223', '321'], 'word': 'p', 'correct': 1},
    {'stim': ['132', '223', '321'], 'word': 'q', 'correct': 2, 'pragmatics_wrong': '223'},
    {'stim': ['233', '121', '111'], 'word': 'p', 'correct': 1},
    {'stim': ['332', '313', '311'], 'word': 'q', 'correct': 2},
]
stimulus_listener_3 = [
    {'stim': ['232', '312', '322'], 'word': 'q', 'correct': 1},
    {'stim': ['121', '111', '232'], 'word': 'p', 'correct': 2},
    {'stim': ['112', '121', '221'], 'word': 'q', 'correct': 0},
    {'stim': ['112', '121', '221'], 'word': 'q', 'correct': 0},
    {'stim': ['112', '121', '221'], 'word': 'p', 'correct': 2, 'pragmatics_wrong': '112'},
    {'stim': ['212', '222', '232'], 'word': 'q', 'correct': 0},
    {'stim': ['133', '132', '231'], 'word': 'p', 'correct': 2},
    {'stim': ['211', '113', '133'], 'word': 'p', 'correct': 0},
    {'stim': ['211', '113', '133'], 'word': 'q', 'correct': 1, 'pragmatics_wrong': '211'},
    {'stim': ['321', '213', '131'], 'word': 'q', 'correct': 1},
    {'stim': ['311', '112', '131'], 'word': 'q', 'correct': 1},
    {'stim': ['311', '112', '131'], 'word': 'p', 'correct': 0},
    {'stim': ['132', '133', '333'], 'word': 'p', 'correct': 2},
    {'stim': ['313', '112', '133'], 'word': 'p', 'correct': 0},
    {'stim': ['112', '312', '113'], 'word': 'p', 'correct': 1},
    {'stim': ['111', '221', '313'], 'word': 'q', 'correct': 2},
    {'stim': ['111', '221', '313'], 'word': 'p', 'correct': 1, 'pragmaticswrongh': '313'},
    {'stim': ['323', '231', '212'], 'word': 'q', 'correct': 2},
    {'stim': ['122', '233', '133'], 'word': 'p', 'correct': 1},
    {'stim': ['312', '333', '121'], 'word': 'q', 'correct': 0},
    {'stim': ['312', '333', '121'], 'word': 'p', 'correct': 1, 'pragmaticswrongh': '312'},
    {'stim': ['311', '111', '213'], 'word': 'q', 'correct': 2},
]
stimulus_listener_4 = [
    {'stim': ['311', '331', '123'], 'word': 'p', 'correct': 0},
    {'stim': ['321', '221', '113'], 'word': 'q', 'correct': 0},
    {'stim': ['111', '312', '212'], 'word': 'q', 'correct': 1},
    {'stim': ['111', '212', '312'], 'word': 'q', 'correct': 2},
    {'stim': ['111', '212', '312'], 'word': 'p', 'correct': 0, 'pragmatics_wrong': '312'},
    {'stim': ['212', '213', '211'], 'word': 'p', 'correct': 2},
    {'stim': ['231', '131', '311'], 'word': 'p', 'correct': 2},
    {'stim': ['221', '131', '211'], 'word': 'p', 'correct': 2},
    {'stim': ['221', '131', '211'], 'word': 'q', 'correct': 1, 'pragmatics_wrong': '211'},
    {'stim': ['333', '121', '212'], 'word': 'q', 'correct': 0},
    {'stim': ['311', '223', '112'], 'word': 'q', 'correct': 0},
    {'stim': ['223', '232', '111'], 'word': 'p', 'correct': 2},
    {'stim': ['223', '232', '111'], 'word': 'q', 'correct': 1, 'pragmatics_wrong': '111'},
    {'stim': ['312', '131', '211'], 'word': 'p', 'correct': 2},
    {'stim': ['313', '123', '211'], 'word': 'p', 'correct': 2},
    {'stim': ['313', '221', '222'], 'word': 'q', 'correct': 0},
    {'stim': ['213', '122', '231'], 'word': 'q', 'correct': 2},
    {'stim': ['311', '333', '222'], 'word': 'p', 'correct': 0},
    {'stim': ['311', '333', '222'], 'word': 'p', 'correct': 0},
    {'stim': ['311', '333', '222'], 'word': 'q', 'correct': 1, 'pragmaticswrongh': '311'},
    {'stim': ['123', '111', '133'], 'word': 'q', 'correct': 2},
    {'stim': ['132', '323', '111'], 'word': 'p', 'correct': 2},
]

# Create hypotheses for each concept (they are the same in the beginning) and the lexica.
p_hypotheses, q_hypotheses = create_hypotheses_space()
lexica = create_lexica(p_hypotheses, q_hypotheses)

list_lexica = []
list_p_hypotheses = []
list_q_hypotheses = []
# These dataframes will store best lexica, p_hypotheses and q_hypotheses, for each block
df_lexica = pd.DataFrame(columns=['Block', 'Index', 'p', 'q', 'Probability'])
df_p_hypotheses = pd.DataFrame(columns=['Block', 'Index', 'Hypothesis', 'Probability'])
df_q_hypotheses = pd.DataFrame(columns=['Block', 'Index', 'Hypothesis', 'Probability'])

stimuli_listener = [stimulus_listener_1, stimulus_listener_2, stimulus_listener_3, stimulus_listener_4]

prob_correct = run_experiment(4,0,0.525)
# The following line is to compare the results to those of the control group (the order of trials is rearranged)
#prob_correct = run_experiment_control()

df_prob_correct = pd.DataFrame.from_dict(prob_correct, orient='index')
df_prob_correct = df_prob_correct.transpose()
df_prob_correct.to_csv('combinedmodel.csv', index=False)
df_lexica = pd.concat([df_lexica, pd.DataFrame(list_lexica)], ignore_index=True)
df_p_hypotheses = pd.concat([df_p_hypotheses, pd.DataFrame(list_p_hypotheses)], ignore_index=True)
df_q_hypotheses = pd.concat([df_q_hypotheses, pd.DataFrame(list_q_hypotheses)], ignore_index=True)
df_lexica.to_csv('best_lexica.csv', index=False)
df_p_hypotheses.to_csv('best_p_hypotheses.csv', index=False)
df_q_hypotheses.to_csv('best_q_hypotheses.csv', index=False)
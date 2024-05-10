import pandas as pd
import matplotlib.pyplot as plt

models = {
    "Double Update Model": pd.read_csv("pr0.525.csv"),
    "Control Group Model": pd.read_csv("probability of correct responses control group lot only.csv"),
    "Basic Model": pd.read_csv("probability of correct responses experimental group lot only.csv"),
    "L2 Model": pd.read_csv("a4c0.csv")
} 
experiment_probres = pd.read_csv("experiment_probres.csv")
experiment_probrescontrol = pd.read_csv("experiment_probrescontrol.csv")
experimental_results = experiment_probres['correct']
control_group_experimental_results = experiment_probrescontrol['correct']
blocks = experiment_probres['block']

def plot_with_dual_yaxes(experimental_results, models_results, blocks, m_label_modifications, q_label_modifications, title, y_min = 0.20, y_max = 0.55):
    x_axis = list(range(1, len(experimental_results) + 1))

    fig, ax1 = plt.subplots(figsize=(12, 6))
    #plt.title(title)

    # Plot experimental results on the first y-axis
    ax1.plot(x_axis, experimental_results, label='Experimental Results', linestyle='-', marker='o', color='b')
    ax1.set_ylabel("Participants' Probability of Responding Correctly", color='b')
    ax1.set_ylim(-0.05, 1.05)

    # Create a second y-axis for the model results
    ax2 = ax1.twinx()
    ax2.plot(x_axis, models_results, label='Models Results', linestyle='--', marker='x', color='r')
    ax2.set_ylabel("Model's Probability of Responding Correctly", color='r')
    ax2.set_ylim(y_min, y_max)

    # Highlight pragmatic trials
    def highlight_points(label_modifications, data, color, ax):
        for block, indices in label_modifications.items():
            for index in indices:
                experimental_index = sum(1 for b in blocks if b < block) + index
                ax.plot(experimental_index + 1, data[experimental_index], marker='s', color=color, linestyle='', markersize=10)

    highlight_points(m_label_modifications, experimental_results, 'b', ax1)
    highlight_points(m_label_modifications, models_results, 'r', ax2)
    highlight_points(q_label_modifications, experimental_results, 'b', ax1)
    highlight_points(q_label_modifications, models_results, 'r', ax2)

    # Add vertical lines for block transitions
    block_change_indices = blocks[blocks.diff() != 0].index.tolist()
    for idx in block_change_indices:
        ax1.axvline(x=idx + 1, linestyle='--', color='grey', alpha=0.5)

    # Label blocks on the x-axis
    block_labels = {idx + 1: f"Block {int(blocks[idx + 1])}" for idx in block_change_indices}
    ax1.set_xticks(list(block_labels.keys()))
    ax1.set_xticklabels(list(block_labels.values()))  # Label block numbers without decimals

    plt.xlabel("Task Progression")

    plt.show()


# Define the label modifications for the implicatures
m_imp_experimental = {
    1: [4, 8, 14],
    2: [5, 10, 14, 19],
    3: [4, 8],
    4: [4, 8, 12],
}
q_imp_experimental = {
    1: [19],
    2: [],
    3: [16, 20],
    4: [19],
}
# In control group, M- and Q-implicatures are not really distinguished
m_imp_control = {
    1: [3, 9, 13, 17],
    2: [2, 3, 18, 20],
    3: [1, 8, 10, 15],
    4: [1, 14, 17, 20],
}
q_imp_control = {
    1: [],
    2: [],
    3: [],
    4: [],
}            

for model_name, df in models.items():
    models[model_name] = pd.concat([df[str(i)] for i in range(1, 5)], ignore_index=True)
    
for model_name, model_results in models.items():
    title = f"Experimental Results vs {model_name}"
    if model_name == "Basic Model":
        plot_with_dual_yaxes(
            experimental_results,
            model_results,
            blocks,
            m_imp_experimental,
            q_imp_experimental,
            title
        )
    elif model_name == "Control Group Model":
        plot_with_dual_yaxes(
            control_group_experimental_results,
            model_results,
            blocks,
            m_imp_control,
            q_imp_control,
            title
        )
    elif model_name == "Double Update Model":
        plot_with_dual_yaxes(
            experimental_results,
            model_results,
            blocks,
            m_imp_experimental,
            q_imp_experimental,
            title,
            y_max=0.60
        )
    else:
        plot_with_dual_yaxes(
            experimental_results,
            model_results,
            blocks,
            m_imp_experimental,
            q_imp_experimental,
            title,
            y_min=-0.05,
            y_max=0.85
        )
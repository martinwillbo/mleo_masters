import os
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from support_functions_logging import class_names

def scalars_to_dataframe(log_dir):
    # Initialize an accumulator to collect all scalar data
    ea = event_accumulator.EventAccumulator(log_dir, size_guidance={
        event_accumulator.SCALARS: 0,  # 0 means load all scalars
    })

    # Load the data
    ea.Reload()

    # Prepare DataFrame
    all_scalars_df = pd.DataFrame()

    # Iterate over all scalar tags and read the data
    for tag in ea.Tags()['scalars']:
        scalar_events = ea.Scalars(tag)
        df = pd.DataFrame([(e.step, e.value) for e in scalar_events], columns=['Step', tag])
        if all_scalars_df.empty:
            all_scalars_df = df
        else:
            all_scalars_df = pd.merge(all_scalars_df, df, on='Step', how='outer')

    return all_scalars_df

def plot_histogram(text_values, log_dir, title="Histogram of Text Values"):
    # Convert text values to numerical values
    bin_edges = list(range(len(text_values)))
    
    # Plotting the histogram
    plt.figure(figsize=(10, 8))
    plt.bar(bin_edges, text_values, width=1.0, align='edge')
    plt.title(title)
    plt.xlabel('Class')
    plt.ylabel('mIoU')
    plt.xticks(bin_edges, [str(i) for i in bin_edges]) 
    plt.grid(True)
    plt.savefig(f'{log_dir}/{title}.png') 
    plt.close()  # Close the figure to free up memory

def plot_2_histograms(values_LUPI, values_wo, log_dir, title):
    n_classes = len(values_LUPI)

    # Setting up the positions for the bars
    index = np.arange(n_classes) *1.5  # Class indices
    bar_width = 0.35  # Width of the bars

    # Creating the bar plot
    plt.figure(figsize=(16, 8))

    # Plot bars for the first group
    plt.bar(index, values_LUPI, bar_width, label='Trained with privileged information')

    # Plot bars for the second group, offset by the width of the bars
    plt.bar(index + bar_width, values_wo, bar_width, label='Trained without privileged informaion')

    # Customize the plot
    plt.title('mIoU per class')
    plt.xlabel('Class')
    plt.ylabel('mIoU')
    plt.xticks(index + bar_width / 2, [str(i+1) for i in range(n_classes)])  # Update class labels and font size
    plt.legend()

    textstr = '\n'.join([f'{i+1}: {name}' for i, name in enumerate(class_names)])  # Example with a few class names
    plt.annotate(textstr, xy=(1.02, 0.5), xycoords='axes fraction', fontsize=10,
             bbox=dict(boxstyle="round, pad=0.4", fc="lightgray", ec="black", lw=1))

    plt.grid(True)
    plt.savefig(f'{log_dir}/{title}.png', bbox_inches='tight') 
    plt.close()  # Close the figure to free up memory

    
def handle_df(log_dir):    
    # Convert all scalar data to a pandas DataFrame
    scalar_data_df = scalars_to_dataframe(log_dir)

    # Display the first few rows of the DataFrame
    print(scalar_data_df.shape)

    #search for min loss to find the epoch to use on test etc
    min_val_loss_idx = scalar_data_df['val/loss'].idxmin()
    val_columns = [col for col in scalar_data_df.columns if col.startswith('val/')]
    filtered_df = scalar_data_df[val_columns]
    min_val_loss_row = filtered_df.loc[min_val_loss_idx]

    # Step 5: Print the values
    print(min_val_loss_row)
    print(f"Row with minimum 'val/loss': {min_val_loss_idx}")

# Directory where your TensorBoard logs are stored
model_folder = '2024-01-29_18-13-26'

log_dir = '../../'+model_folder+'/tensorboard'

#extracted manually
base_values_wo = [0.85633034, 0.7118823, 0.79254436, 0.79795754, 0.8797694, 0.6296923, 0.7697543, 0.57710105, 0.8872009, 0.6940321, 0.82619613, 0.8196255, 0.5436313, 0.88608176, 0.46711946, 0.7061641, 0.58820695, 0.8491268, 0.12912768]
base_values_LUPI = [0.87521106, 0.7419038, 0.81973195, 0.83298403, 0.9133609, 0.66294664, 0.79159534, 0.60244775, 0.8982074, 0.7144704, 0.8457025, 0.8352307, 0.591786, 0.90839034, 0.7514763, 0.58104175, 0.590506, 0.88457185, 0.19094262]
base_values_wo_5 = [0.75881314, 0.49890247, 0.68694746, 0.5373044, 0.7757337, 0.4352869, 0.6926766, 0.43489882, 0.8411076, 0.5060508, 0.5859925, 0.49910438, 0.3463579, 0.29609293, 0.23242086, 0.0, 0.0, 0.004436234, 0.0]
base_values_LUPI_5 = [0.8164449, 0.5189286, 0.7087663, 0.6449882, 0.84200996, 0.4554332, 0.702085, 0.4009663, 0.80855113, 0.51986355, 0.59820104, 0.71825534, 0.42632073, 0.22271551, 0.2219956, 0.0, 0.0, 0.02705991, 0.0]

plot_histogram(base_values_wo, log_dir, title="Histogram of mIoU per Class at Min Val Loss Epoch")

title_5="Histogram of mIoU per class for model trained with and without privileged information, 5 percent of dataset"
plot_2_histograms(base_values_LUPI_5, base_values_wo_5, log_dir, title_5)
plot_2_histograms(base_values_LUPI, base_values_wo, log_dir, title="mIoU_hist_base_100")




def plot_function(scalar_data_df):
    plots_dir = '../../'+model_folder+'/plots'
    columns_to_plot = scalar_data_df.columns[1:]
    for column in columns_to_plot:
        print(f'{plots_dir}{column}.png')
        plt.figure()
        plt.plot(scalar_data_df['step'], scalar_data_df[column], label=column)
        plt.xlabel('Step')
        plt.ylabel(column)
        plt.title(f'{column} over Steps')
        plt.legend()
        plt.savefig(f'{plots_dir}{column}.png')  # Save the figure with the column name
        plt.close()  # Close the figure to free up memory
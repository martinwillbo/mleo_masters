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
model_folder = '2024-02-08_13-51-52'

log_dir = '../../logs/'+model_folder+'/tensorboard'

#extracted manually
base_values_wo = [0.8038167, 0.6085486, 0.7509828, 0.636876, 0.81301343, 0.39043948, 0.64369786, 0.40655535, 0.7757048, 0.5586019, 0.6027413, 0.67422533, 0.5716616, 0.8249599, 0.0028337273, 4.7015596e-06, 0.06651074, 0.27852723, 0.038257338]
#base_values_LUPI = [0.87521106, 0.7419038, 0.81973195, 0.83298403, 0.9133609, 0.66294664, 0.79159534, 0.60244775, 0.8982074, 0.7144704, 0.8457025, 0.8352307, 0.591786, 0.90839034, 0.7514763, 0.58104175, 0.590506, 0.88457185, 0.19094262]
#base_values_wo_5 = [0.75279707, 0.5507048, 0.69714856, 0.57836574, 0.8280502, 0.25952816, 0.5512302, 0.41607723, 0.69347316, 0.49848855, 0.5443956, 0.6938274, 0.4766166, 0.47294053, 0.0, 0.0, 0.00074471155, 0.3588589, 0.029811265]
#base_values_LUPI_5 = [0.8146344, 0.5885024, 0.7245423, 0.6366687, 0.82678473, 0.2947062, 0.64695907, 0.44606584, 0.66489345, 0.54620415, 0.54704773, 0.6078559, 0.5312133, 0.7371452, 0.0, 3.387229e-05, 0.004924396, 0.5028451, 0.005206798]

#plot_histogram(base_values_wo, log_dir, title="Histogram of mIoU per Class at Min Val Loss Epoch")

#title_5="Histogram of mIoU per class for model trained with and without privileged information, 5 percent of dataset"
plot_2_histograms(base_values_LUPI_5, base_values_wo_5, log_dir, title="mIoU_hist_base_5")
#plot_2_histograms(base_values_LUPI, base_values_wo, log_dir, title="mIoU_hist_base_100")

#handle_df(log_dir)




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
import os
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import matplotlib.pyplot as plt
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
# Directory where your TensorBoard logs are stored
model_folder = 'best_model_5_channels_5'
log_dir = '../log_res/'+model_folder+'/tensorboard'
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
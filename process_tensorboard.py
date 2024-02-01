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
model_folder = '2024-01-29_18-13-26/'

log_dir = '../../'+model_folder+'/tensorboard'
plots_dir = '../../'+model_folder+'/plots'

# Convert all scalar data to a pandas DataFrame
scalar_data_df = scalars_to_dataframe(log_dir)

# Display the first few rows of the DataFrame
print(scalar_data_df.shape)

columns_to_plot = scalar_data_df.columns[1:]


for column in columns_to_plot:
    print(f'{plots_dir}{column}.png')
    # plt.figure()
    # plt.plot(scalar_data_df['step'], scalar_data_df[column], label=column)
    # plt.xlabel('Step')
    # plt.ylabel(column)
    # plt.title(f'{column} over Steps')
    # plt.legend()
    # plt.savefig(f'{plots_dir}{column}.png')  # Save the figure with the column name
    # plt.close()  # Close the figure to free up memory
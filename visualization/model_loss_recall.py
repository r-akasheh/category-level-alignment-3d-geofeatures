import pandas as pd
import matplotlib.pyplot as plt

item_type = "teapot"
# File paths to your CSV files
loss_csv_path = 'C:/master/robot-vision-modul/report/figures/train-loss-csv/' + item_type + ".csv"
accuracy_csv_path = 'C:/master/robot-vision-modul/report/figures/train-recall-csv/' + item_type + ".csv"
recall_val_csv = 'C:/master/robot-vision-modul/report/figures/val-recall-csv/' + item_type + ".csv"

# Load CSV files into DataFrames
loss_df = pd.read_csv(loss_csv_path)
accuracy_df = pd.read_csv(accuracy_csv_path)
recall_val_df = pd.read_csv(recall_val_csv)

# Create a figure
fig, axs = plt.subplots(1, 3, figsize=(15, 7))

# Plot Loss
axs[0].plot(loss_df['Step'], loss_df['Value'], label='Loss', linewidth=2.5)
axs[0].set_xlabel('Step')
axs[0].set_ylabel('Loss')
axs[0].set_title('Training Loss')
axs[0].legend()
axs[0].grid(False)

# Plot Accuracy
axs[1].plot(accuracy_df['Step'], accuracy_df['Value'], label='Recall', color='orange', linewidth=2.5)
axs[1].set_xlabel('Step')
axs[1].set_ylabel('Recall')
axs[1].set_title('Training Recall')
axs[1].legend()
axs[1].grid(False)

# Plot Accuracy
axs[2].plot(recall_val_df['Step'], recall_val_df['Value'], label='Recall', color='red', linewidth=2.5)
axs[2].set_xlabel('Step')
axs[2].set_ylabel('Validation Recall')
axs[2].set_title('Validation Recall')
axs[2].legend()
axs[2].grid(False)

# Adjust layout
plt.tight_layout()

# Save the figure
# plt.savefig('test1.png')

# Show the plot
plt.show()

print("Plots saved as 'custom_plots.png'")

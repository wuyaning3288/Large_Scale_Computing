import pandas as pd
import tensorflow as tf

# Load the dataset
train_X = pd.read_csv('train_X.csv', header=None).values
train_Y = pd.read_csv('train_Y.csv', header=None).values

# Define the ANN model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(train_X.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(train_Y.shape[1], activation='linear')  # Linear activation for regression
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model using validation_split
history = model.fit(
    train_X, train_Y,
    epochs=1000,  # Equivalent to 10000 passes over the data
    batch_size=5000,  # Batch size of 5000
    validation_split=0.01,  # Use 1% of data as validation set
    verbose=1  # Print training progress
)

# Save the trained model
model.save('three_body_model.h5')
# Plot training and validation loss (MAE) vs epochs
def plot_mae(history, output_file='mae_vs_epoch.png'):
 
    # Extract MAE from training history
    train_mae = history.history['mae']
    val_mae = history.history['val_mae']
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_mae, label='Training MAE', color='blue', linestyle='solid')
    plt.plot(val_mae, label='Validation MAE', color='orange', linestyle='dashed')
    plt.title('Mean Absolute Error (MAE) vs Epoch', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Mean Absolute Error (MAE)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()

    # Save the plot as a PNG file
    plt.savefig(output_file, format='png', dpi=300)
    plt.close()

# Plot the MAE using the training history
plot_mae(history, output_file='mae_vs_epoch.png')

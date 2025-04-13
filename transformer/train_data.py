# Load the dataset
df = pd.read_csv('synthetic_training_data.csv')

# Convert labels to numeric
label_map = {'normal': 0, 'accident': 1, 'drowsy': 2, 'near_accident': 3, 'rough_road': 4, 'fatigue': 5}
df['label_id'] = df['label'].map(label_map)

# Extract sequences
sequences = []
labels = []

for seq_id in df['sequence_id'].unique():
    seq_data = df[df['sequence_id'] == seq_id]
    
    # Extract features
    features = seq_data[['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']].values
    
    # Calculate additional features
    acc_mag = np.linalg.norm(seq_data[['acc_x', 'acc_y', 'acc_z']].values, axis=1)
    gyro_mag = np.linalg.norm(seq_data[['gyro_x', 'gyro_y', 'gyro_z']].values, axis=1)
    
    # Calculate jerk (derivative of acceleration)
    acc_jerk = np.gradient(features[:, 0:3], axis=0)
    acc_jerk_mag = np.linalg.norm(acc_jerk, axis=1)
    
    # Add engineered features
    full_features = np.column_stack([
        features, 
        acc_mag, 
        gyro_mag, 
        acc_jerk_mag
    ])
    
    # Get primary label for the sequence (most frequent non-normal label)
    label_counts = seq_data['label'].value_counts()
    if len(label_counts) == 1 and label_counts.index[0] == 'normal':
        label = 'normal'
    else:
        # Remove 'normal' from counts if it exists
        if 'normal' in label_counts:
            label_counts = label_counts.drop('normal')
        label = label_counts.index[0]
    
    sequences.append(full_features)
    labels.append(label_map[label])

# Convert to numpy arrays
sequences = np.array(sequences)
labels = np.array(labels)

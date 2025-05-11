import numpy as np
import pandas as pd
import nibabel as nib
import os
from sklearn.model_selection import train_test_split

# === Step 1: Paths ===
main_folder = r'home\linga011'
excel_file = r'home\linga011\name_mapping_pre-processing.xlsx'

# === Step 2: Load preprocessing Excel file ===
df = pd.read_excel(excel_file)

# Strip any extra spaces from column names
df.columns = df.columns.str.strip()

# Rename columns for easier access (optional)
df.rename(columns={'Starting slice': 'StartSlice', 'End slice': 'EndSlice'}, inplace=True)

# === Step 3: List patient folders ===
patient_folders = sorted([f for f in os.listdir(main_folder) if f.startswith('BraTS20_Training_')])

print(f"Found {len(patient_folders)} patient folders!")


# === Step 4: Load mean images and filter valid patients ===

mean_images = []
labels = []
start_slices = []
end_slices = []
valid_patient_ids = []  # for tracking

for folder in patient_folders:
    mean_file_path = os.path.join(main_folder, folder, f'{folder}_mean1.nii.gz')
    
    if os.path.exists(mean_file_path):
        img = nib.load(mean_file_path)
        img_data = img.get_fdata()  # (160, 210, 100)

        pid = folder.split('_')[-1]
        full_id = f'BraTS20_Training_{pid}'

        entry = df[df['BraTS_2020_subject_ID'] == full_id]

        if not entry.empty:
            grade = entry['Grade'].values[0]
            start = entry['StartSlice'].values[0]
            end = entry['EndSlice'].values[0]

            # === New: Check for NaN or bad text ===
            if (isinstance(start, (int, float)) and isinstance(end, (int, float))
                and not np.isnan(start) and not np.isnan(end)):
                
                label = 1 if grade == 'HGG' else 0
                labels.append(label)
                start_slices.append(int(start))
                end_slices.append(int(end))

                mean_images.append(img_data)
                valid_patient_ids.append(pid)
            else:
                print(f" Skipping patient {pid}: Invalid StartSlice/EndSlice ({start})")
        else:
            print(f" Patient {pid} not found in Excel!")
    else:
        print(f" Mean file missing for {folder}")

# Final stack after filtering
mean_images = np.stack(mean_images, axis=0)
labels = np.array(labels)
start_slices = np.array(start_slices)
end_slices = np.array(end_slices)

print(f" Loaded mean images: {mean_images.shape}")
print(f" Loaded labels: {labels.shape}")



# === Step 6: Train-Test Split (70:30 stratified by label) ===
train_idx, test_idx = train_test_split(np.arange(len(mean_images)),
                                       test_size=0.3,
                                       random_state=42,
                                       stratify=labels)

print(f" Split complete: {len(train_idx)} training patients, {len(test_idx)} testing patients.")

# === Step 7: Extract 2D slices ===

def extract_slices(patient_indices):
    slices = []
    slice_labels = []
    for idx in patient_indices:
        vol = mean_images[idx]  # (160,210,100)
        start = start_slices[idx]
        end = end_slices[idx]
        
        selected_slices = vol[:, :, start:end+1]  # Extract slices between start and end (inclusive)
        
        for i in range(selected_slices.shape[-1]):
            slice_2d = selected_slices[:, :, i]  # Shape: (160,210)
            slices.append(slice_2d)
            slice_labels.append(labels[idx])  # Same label for all slices of this patient
    
    slices = np.array(slices)
    slice_labels = np.array(slice_labels)
    return slices, slice_labels

# Extract for training
train_slices, train_labels = extract_slices(train_idx)
# Extract for testing
test_slices, test_labels = extract_slices(test_idx)

print(f" Extracted slices -> Train: {train_slices.shape}, Test: {test_slices.shape}")

# === Step 8: Save all files ===
np.save('Train.npy', train_slices)
np.save('Train_labels.npy', train_labels)
np.save('Test.npy', test_slices)
np.save('Test_labels.npy', test_labels)

print(" Saved Train.npy, Train_labels.npy, Test.npy, Test_labels.npy successfully!")

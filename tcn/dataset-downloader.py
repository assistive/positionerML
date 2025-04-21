"""
Download and prepare public datasets for IMU-based dead reckoning.

This script downloads and processes several public datasets suitable for
training IMU dead reckoning models, including:
1. Oxford Inertial Odometry Dataset (OxIOD)
2. RoNIN Dataset
3. RIDI Dataset
4. KITTI Dataset segments with IMU data

Usage:
  python download_datasets.py --dataset oxiod --output_dir ./data
"""

import os
import sys
import argparse
import zipfile
import tarfile
import shutil
import urllib.request
import pandas as pd
import numpy as np
from tqdm import tqdm
import h5py


class ProgressBar(tqdm):
    """Progress bar for downloads."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url, destination):
    """
    Download a file with a progress bar
    
    Args:
        url: URL to download
        destination: Path to save the file
    """
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    if os.path.exists(destination):
        print(f"File already exists: {destination}")
        return
    
    print(f"Downloading {url} to {destination}")
    with ProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=destination, reporthook=t.update_to)


def extract_archive(archive_path, extract_path):
    """
    Extract a zip or tar archive
    
    Args:
        archive_path: Path to the archive file
        extract_path: Path to extract the contents
    """
    os.makedirs(extract_path, exist_ok=True)
    
    print(f"Extracting {archive_path} to {extract_path}")
    
    if archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            for member in tqdm(zip_ref.infolist(), desc='Extracting'):
                zip_ref.extract(member, extract_path)
    elif archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz'):
        with tarfile.open(archive_path, 'r:gz') as tar_ref:
            for member in tqdm(tar_ref.getmembers(), desc='Extracting'):
                tar_ref.extract(member, extract_path)
    elif archive_path.endswith('.tar'):
        with tarfile.open(archive_path, 'r:') as tar_ref:
            for member in tqdm(tar_ref.getmembers(), desc='Extracting'):
                tar_ref.extract(member, extract_path)
    else:
        print(f"Unsupported archive format: {archive_path}")


def download_oxiod(output_dir):
    """
    Download the Oxford Inertial Odometry Dataset (OxIOD)
    
    Args:
        output_dir: Directory to save the dataset
    """
    # Create output directory
    oxiod_dir = os.path.join(output_dir, 'oxiod')
    os.makedirs(oxiod_dir, exist_ok=True)
    
    # OxIOD dataset URLs
    oxiod_urls = {
        'walking': 'https://drive.google.com/uc?export=download&id=1ZJevyQfOCuzuEb2XQaEfLpAekbKcU-1F',
        'running': 'https://drive.google.com/uc?export=download&id=13qJclkOoHFDlHmS3eAzI_2LQHIQWNa4V',
        'stair': 'https://drive.google.com/uc?export=download&id=1GIkHZqOcCXpXPQ3P2hUTk6g7YZz5pHqC',
    }
    
    # Download and extract each dataset
    for activity, url in oxiod_urls.items():
        zip_path = os.path.join(oxiod_dir, f'{activity}.zip')
        extract_path = os.path.join(oxiod_dir, activity)
        
        # Use gdown for Google Drive downloads
        try:
            import gdown
            gdown.download(url, zip_path, quiet=False)
        except ImportError:
            print("Please install gdown to download Google Drive files:")
            print("pip install gdown")
            return
        
        # Extract the downloaded archive
        extract_archive(zip_path, extract_path)
    
    # Process the extracted data
    process_oxiod(oxiod_dir)


def process_oxiod(oxiod_dir):
    """
    Process the Oxford Inertial Odometry Dataset
    
    Args:
        oxiod_dir: Directory containing the OxIOD dataset
    """
    print("Processing OxIOD dataset...")
    
    # Find all data files
    data_files = []
    for root, dirs, files in os.walk(oxiod_dir):
        for file in files:
            if file.endswith('.csv') and ('synced' in file or 'sync' in file):
                data_files.append(os.path.join(root, file))
    
    print(f"Found {len(data_files)} data files")
    
    # Process each file and convert to standard format
    for file_path in tqdm(data_files, desc='Processing OxIOD files'):
        try:
            # Read data
            df = pd.read_csv(file_path)
            
            # Check column names (they vary in the dataset)
            columns = df.columns.tolist()
            
            # Map columns to standard names
            column_mapping = {}
            
            # Accelerometer columns
            if 'a_x' in columns:
                column_mapping['a_x'] = 'accel_x'
                column_mapping['a_y'] = 'accel_y'
                column_mapping['a_z'] = 'accel_z'
            elif 'accX' in columns:
                column_mapping['accX'] = 'accel_x'
                column_mapping['accY'] = 'accel_y'
                column_mapping['accZ'] = 'accel_z'
            
            # Gyroscope columns
            if 'g_x' in columns:
                column_mapping['g_x'] = 'gyro_x'
                column_mapping['g_y'] = 'gyro_y'
                column_mapping['g_z'] = 'gyro_z'
            elif 'gyroX' in columns:
                column_mapping['gyroX'] = 'gyro_x'
                column_mapping['gyroY'] = 'gyro_y'
                column_mapping['gyroZ'] = 'gyro_z'
            
            # Position columns
            if 'p_x' in columns:
                column_mapping['p_x'] = 'pos_x'
                column_mapping['p_y'] = 'pos_y'
                column_mapping['p_z'] = 'pos_z'
            elif 'x' in columns:
                column_mapping['x'] = 'pos_x'
                column_mapping['y'] = 'pos_y'
                column_mapping['z'] = 'pos_z'
            
            # Rename columns
            df = df.rename(columns=column_mapping)
            
            # Add sequence identifier
            sequence_name = os.path.basename(file_path).split('.')[0]
            df['sequence'] = sequence_name
            
            # Save processed file
            processed_path = file_path.replace('.csv', '_processed.csv')
            df.to_csv(processed_path, index=False)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    print("OxIOD dataset processing complete!")


def download_ronin(output_dir):
    """
    Download the RoNIN Dataset
    
    Args:
        output_dir: Directory to save the dataset
    """
    # Create output directory
    ronin_dir = os.path.join(output_dir, 'ronin')
    os.makedirs(ronin_dir, exist_ok=True)
    
    # RoNIN dataset URLs
    ronin_urls = {
        'ronin_sample': 'https://datasets.iclr.cc/dataset/3b4148a14c0d4bae819e4cad9b8b4e3b/ronin_sample/download',
    }
    
    # Download and extract the dataset
    for part, url in ronin_urls.items():
        zip_path = os.path.join(ronin_dir, f'{part}.zip')
        
        # Download
        download_file(url, zip_path)
        
        # Extract
        extract_archive(zip_path, ronin_dir)
    
    # Process the extracted data
    process_ronin(ronin_dir)


def process_ronin(ronin_dir):
    """
    Process the RoNIN Dataset
    
    Args:
        ronin_dir: Directory containing the RoNIN dataset
    """
    print("Processing RoNIN dataset...")
    
    # Find all data files
    data_files = []
    for root, dirs, files in os.walk(ronin_dir):
        for file in files:
            if file.endswith('.h5'):
                data_files.append(os.path.join(root, file))
    
    print(f"Found {len(data_files)} data files")
    
    # Process each file and convert to standard CSV format
    for file_path in tqdm(data_files, desc='Processing RoNIN files'):
        try:
            # Create output directory for this sequence
            sequence_name = os.path.basename(file_path).split('.')[0]
            output_path = os.path.join(os.path.dirname(file_path), f"{sequence_name}_processed.csv")
            
            # Read h5 file
            with h5py.File(file_path, 'r') as f:
                # Extract data
                if 'synced' in f:
                    # Get accelerometer data
                    accel = f['synced/accel'][...]
                    
                    # Get gyroscope data
                    gyro = f['synced/gyro'][...]
                    
                    # Get timestamps
                    timestamps = f['synced/time'][...]
                    
                    # Get positions (from VIO)
                    if 'pos' in f['synced']:
                        pos = f['synced/pos'][...]
                    else:
                        print(f"No position data in {file_path}")
                        continue
                    
                    # Create DataFrame
                    df = pd.DataFrame({
                        'timestamp': timestamps,
                        'accel_x': accel[:, 0],
                        'accel_y': accel[:, 1],
                        'accel_z': accel[:, 2],
                        'gyro_x': gyro[:, 0],
                        'gyro_y': gyro[:, 1],
                        'gyro_z': gyro[:, 2],
                        'pos_x': pos[:, 0],
                        'pos_y': pos[:, 1],
                        'pos_z': pos[:, 2],
                        'sequence': sequence_name
                    })
                    
                    # Save processed file
                    df.to_csv(output_path, index=False)
                else:
                    print(f"No 'synced' group in {file_path}")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    print("RoNIN dataset processing complete!")


def download_ridi(output_dir):
    """
    Download the RIDI Dataset
    
    Args:
        output_dir: Directory to save the dataset
    """
    # Create output directory
    ridi_dir = os.path.join(output_dir, 'ridi')
    os.makedirs(ridi_dir, exist_ok=True)
    
    # RIDI dataset URLs
    ridi_urls = {
        'data_part1': 'https://drive.google.com/uc?export=download&id=1fWPJjDHm32HeVsBlPxCLPFXWVWIYxWAA',
        'data_part2': 'https://drive.google.com/uc?export=download&id=1dj_bVlwDUTpzD-JXNxoXaI9bnkXw2jEK',
    }
    
    # Download and extract each part
    for part, url in ridi_urls.items():
        zip_path = os.path.join(ridi_dir, f'{part}.zip')
        
        # Use gdown for Google Drive downloads
        try:
            import gdown
            gdown.download(url, zip_path, quiet=False)
        except ImportError:
            print("Please install gdown to download Google Drive files:")
            print("pip install gdown")
            return
        
        # Extract the downloaded archive
        extract_archive(zip_path, ridi_dir)
    
    # Process the extracted data
    process_ridi(ridi_dir)


def process_ridi(ridi_dir):
    """
    Process the RIDI Dataset
    
    Args:
        ridi_dir: Directory containing the RIDI dataset
    """
    print("Processing RIDI dataset...")
    
    # Find all data files
    data_files = []
    for root, dirs, files in os.walk(ridi_dir):
        for file in files:
            if file.endswith('.txt') and 'imu' in file:
                data_files.append(os.path.join(root, file))
    
    print(f"Found {len(data_files)} data files")
    
    # Process each file and convert to standard format
    for imu_file_path in tqdm(data_files, desc='Processing RIDI files'):
        try:
            # Find corresponding ground truth file
            gt_file_path = imu_file_path.replace('imu', 'groundtruth')
            if not os.path.exists(gt_file_path):
                print(f"Ground truth file not found for {imu_file_path}")
                continue
            
            # Read IMU data
            imu_data = np.loadtxt(imu_file_path)
            
            # Read ground truth data
            gt_data = np.loadtxt(gt_file_path)
            
            # Create DataFrame
            sequence_name = os.path.basename(imu_file_path).split('.')[0]
            
            # Align lengths
            min_len = min(len(imu_data), len(gt_data))
            
            # Create DataFrame with aligned data
            df = pd.DataFrame({
                'timestamp': imu_data[:min_len, 0],
                'accel_x': imu_data[:min_len, 1],
                'accel_y': imu_data[:min_len, 2],
                'accel_z': imu_data[:min_len, 3],
                'gyro_x': imu_data[:min_len, 4],
                'gyro_y': imu_data[:min_len, 5],
                'gyro_z': imu_data[:min_len, 6],
                'pos_x': gt_data[:min_len, 1],
                'pos_y': gt_data[:min_len, 2],
                'pos_z': gt_data[:min_len, 3],
                'sequence': sequence_name
            })
            
            # Save processed file
            processed_path = imu_file_path.replace('.txt', '_processed.csv')
            df.to_csv(processed_path, index=False)
            
        except Exception as e:
            print(f"Error processing {imu_file_path}: {e}")
    
    print("RIDI dataset processing complete!")


def download_kitti(output_dir):
    """
    Download the KITTI Dataset
    
    Args:
        output_dir: Directory to save the dataset
    """
    # Create output directory
    kitti_dir = os.path.join(output_dir, 'kitti')
    os.makedirs(kitti_dir, exist_ok=True)
    
    # KITTI dataset URLs
    kitti_urls = {
        'data_odometry_raw': 'https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0001/2011_09_26_drive_0001_sync.zip',
        'data_odometry_calib': 'https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_calib.zip',
    }
    
    # Download and extract each part
    for part, url in kitti_urls.items():
        zip_path = os.path.join(kitti_dir, f'{part}.zip')
        
        # Download
        download_file(url, zip_path)
        
        # Extract
        extract_archive(zip_path, kitti_dir)
    
    # Process the extracted data
    process_kitti(kitti_dir)


def process_kitti(kitti_dir):
    """
    Process the KITTI Dataset
    
    Args:
        kitti_dir: Directory containing the KITTI dataset
    """
    print("Processing KITTI dataset...")
    
    # Find the oxts data directory
    oxts_dirs = []
    for root, dirs, files in os.walk(kitti_dir):
        if 'oxts' in root:
            oxts_dirs.append(root)
    
    if not oxts_dirs:
        print("No oxts data found in KITTI dataset")
        return
    
    # Process each oxts directory
    for oxts_dir in oxts_dirs:
        try:
            # Find data files
            data_files = sorted([os.path.join(oxts_dir, f) for f in os.listdir(oxts_dir) if f.endswith('.txt')])
            
            if not data_files:
                print(f"No data files found in {oxts_dir}")
                continue
            
            print(f"Processing {len(data_files)} files from {oxts_dir}")
            
            # Sequence name
            sequence_name = os.path.basename(os.path.dirname(oxts_dir))
            
            # Process all files into a single sequence
            all_data = []
            
            for file_path in data_files:
                # Read data
                data = np.loadtxt(file_path)
                all_data.append(data)
            
            # Combine all data
            combined_data = np.vstack(all_data)
            
            # Create DataFrame
            # KITTI oxts format: https://www.cvlibs.net/datasets/kitti/setup.php
            # First three columns: lat, lon, alt
            # Next three columns: roll, pitch, yaw
            # Columns 7-13: velocity (forward, lateral, vertical) and acceleration (forward, lateral, vertical, total)
            # Columns 14-19: angular velocity (roll, pitch, yaw) and angular acceleration (roll, pitch, yaw)
            
            df = pd.DataFrame({
                'timestamp': np.arange(len(combined_data)),
                'accel_x': combined_data[:, 11],  # forward acceleration
                'accel_y': combined_data[:, 12],  # lateral acceleration
                'accel_z': combined_data[:, 13],  # vertical acceleration
                'gyro_x': combined_data[:, 17],   # roll rate
                'gyro_y': combined_data[:, 18],   # pitch rate
                'gyro_z': combined_data[:, 19],   # yaw rate
                'lat': combined_data[:, 0],       # latitude
                'lon': combined_data[:, 1],       # longitude
                'alt': combined_data[:, 2],       # altitude
                'sequence': sequence_name
            })
            
            # Convert lat/lon to local coordinates
            lat0, lon0 = df['lat'].iloc[0], df['lon'].iloc[0]
            
            # Simple conversion to meters (approximate)
            # More accurate conversion would use a proper coordinate system
            earth_radius = 6371000  # meters
            df['pos_x'] = (df['lon'] - lon0) * np.cos(lat0 * np.pi/180) * earth_radius * np.pi/180
            df['pos_y'] = (df['lat'] - lat0) * earth_radius * np.pi/180
            df['pos_z'] = df['alt'] - df['alt'].iloc[0]
            
            # Save processed file
            processed_path = os.path.join(kitti_dir, f"{sequence_name}_processed.csv")
            df.to_csv(processed_path, index=False)
            
        except Exception as e:
            print(f"Error processing {oxts_dir}: {e}")
    
    print("KITTI dataset processing complete!")


def combine_datasets(output_dir):
    """
    Combine all processed datasets into a single file
    
    Args:
        output_dir: Directory containing the processed datasets
    """
    print("Combining datasets...")
    
    # Find all processed CSV files
    csv_files = []
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file.endswith('_processed.csv'):
                csv_files.append(os.path.join(root, file))
    
    print(f"Found {len(csv_files)} processed files")
    
    # Combine all files
    all_data = []
    
    for file_path in tqdm(csv_files, desc='Combining datasets'):
        try:
            df = pd.read_csv(file_path)
            all_data.append(df)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    if not all_data:
        print("No data to combine")
        return
    
    # Combine all DataFrames
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Save combined file
    combined_path = os.path.join(output_dir, "combined_dataset.csv")
    combined_df.to_csv(combined_path, index=False)
    
    print(f"Combined dataset saved to {combined_path}")
    print(f"Total samples: {len(combined_df)}")


def main():
    parser = argparse.ArgumentParser(description='Download and process datasets for IMU dead reckoning')
    parser.add_argument('--dataset', type=str, required=True, 
                        choices=['oxiod', 'ronin', 'ridi', 'kitti', 'all'],
                        help='Dataset to download')
    parser.add_argument('--output_dir', type=str, default='./data',
                        help='Directory to save the datasets')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Download requested dataset
    if args.dataset == 'oxiod' or args.dataset == 'all':
        download_oxiod(args.output_dir)
    
    if args.dataset == 'ronin' or args.dataset == 'all':
        download_ronin(args.output_dir)
    
    if args.dataset == 'ridi' or args.dataset == 'all':
        download_ridi(args.output_dir)
    
    if args.dataset == 'kitti' or args.dataset == 'all':
        download_kitti(args.output_dir)
    
    # Combine datasets if requested all
    if args.dataset == 'all':
        combine_datasets(args.output_dir)


if __name__ == "__main__":
    main()

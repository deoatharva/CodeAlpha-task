import os
import numpy as np
import tensorflow as tf
from utils import logger, save_label_mapping

def create_directories():
    """Create the standard project dataset folder structure."""
    dirs = [
        "dataset/MNIST",
        "dataset/EMNIST",
        "dataset/A_Z",
        "dataset/IAM",
        "saved_models",
        "outputs"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        logger.info(f"Directory verified: {d}")

def download_mnist():
    """Downloads MNIST digits dataset and saves locally as .npz"""
    logger.info("Starting MNIST digits download...")
    filepath = "dataset/MNIST/mnist.npz"
    if os.path.exists(filepath):
        logger.info("MNIST dataset already exists locally. Skipping download.")
        return
        
    try:
        # Load standard MNIST dataset
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        np.savez_compressed(filepath, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
        logger.info(f"MNIST digits saved successfully to {filepath}")
        
        # Save mapping
        mnist_map = {i: str(i) for i in range(10)}
        save_label_mapping(mnist_map, "dataset/MNIST/mnist_labels.json")
    except Exception as e:
        logger.error(f"Failed to download MNIST: {e}")

def download_emnist_zip_manually():
    """Manually downloads the EMNIST zip archive directly to the package cache folder."""
    import requests
    cache_dir = os.path.expanduser("~/.cache/emnist")
    os.makedirs(cache_dir, exist_ok=True)
    zip_path = os.path.join(cache_dir, "emnist.zip")
    
    # Check if a valid emnist.zip already exists (greater than 100MB)
    if os.path.exists(zip_path) and os.path.getsize(zip_path) > 100 * 1024 * 1024:
        logger.info("A valid emnist.zip already exists in cache. Skipping download.")
        return True
        
    urls = [
        "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip",
        "https://www.cs.umd.edu/~greg/emnist.zip"
    ]
    
    logger.info("Attempting manual EMNIST download from official HTTPS mirrors...")
    for url in urls:
        try:
            logger.info(f"Downloading EMNIST from mirror: {url}...")
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                total_size = int(response.headers.get('content-length', 0))
                block_size = 1024 * 1024  # 1MB
                downloaded = 0
                with open(zip_path, 'wb') as f:
                    for chunk in response.iter_content(block_size):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"Download Progress: {percent:.2f}% ({downloaded / (1024*1024):.1f}MB / {total_size / (1024*1024):.1f}MB)", end="\r")
                print()
                logger.info("EMNIST download completed successfully.")
                return True
            else:
                logger.warning(f"Mirror returned status code {response.status_code}")
        except Exception as e:
            logger.warning(f"Failed download from mirror {url}: {e}")
            if os.path.exists(zip_path):
                try:
                    os.remove(zip_path)
                except Exception:
                    pass
    return False

def parse_idx_images(file_obj):
    """Parses raw IDX images file and returns a numpy array."""
    magic = int.from_bytes(file_obj.read(4), 'big')
    num_images = int.from_bytes(file_obj.read(4), 'big')
    rows = int.from_bytes(file_obj.read(4), 'big')
    cols = int.from_bytes(file_obj.read(4), 'big')
    
    buf = file_obj.read(num_images * rows * cols)
    data = np.frombuffer(buf, dtype=np.uint8)
    return data.reshape(num_images, rows, cols)

def parse_idx_labels(file_obj):
    """Parses raw IDX labels file and returns a numpy array."""
    magic = int.from_bytes(file_obj.read(4), 'big')
    num_items = int.from_bytes(file_obj.read(4), 'big')
    
    buf = file_obj.read(num_items)
    data = np.frombuffer(buf, dtype=np.uint8)
    return data

def extract_and_parse_emnist_balanced(zip_path):
    """Extracts and parses EMNIST Balanced split directly from the cached zip file."""
    import zipfile
    import gzip
    import io
    
    logger.info(f"Extracting EMNIST Balanced from cached {zip_path}...")
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        namelist = z.namelist()
        
        # Find the balanced split files in the zip
        def find_file(sub):
            for name in namelist:
                if sub in name and name.endswith('.gz'):
                    return name
            raise FileNotFoundError(f"Could not find EMNIST file containing: {sub} in zip")
            
        train_img_name = find_file("balanced-train-images")
        train_lbl_name = find_file("balanced-train-labels")
        test_img_name = find_file("balanced-test-images")
        test_lbl_name = find_file("balanced-test-labels")
        
        logger.info("Parsing training images...")
        with gzip.open(io.BytesIO(z.read(train_img_name)), 'rb') as f:
            x_train = parse_idx_images(f)
            
        logger.info("Parsing training labels...")
        with gzip.open(io.BytesIO(z.read(train_lbl_name)), 'rb') as f:
            y_train = parse_idx_labels(f)
            
        logger.info("Parsing test images...")
        with gzip.open(io.BytesIO(z.read(test_img_name)), 'rb') as f:
            x_test = parse_idx_images(f)
            
        logger.info("Parsing test labels...")
        with gzip.open(io.BytesIO(z.read(test_lbl_name)), 'rb') as f:
            y_test = parse_idx_labels(f)
            
    return x_train, y_train, x_test, y_test

def download_emnist():
    """Downloads EMNIST Balanced split and saves locally as .npz"""
    logger.info("Starting EMNIST Balanced download...")
    filepath = "dataset/EMNIST/emnist_balanced.npz"
    if os.path.exists(filepath):
        logger.info("EMNIST balanced dataset already exists locally. Skipping download.")
        return
        
    # Pre-emptively download zip to cache if missing or corrupt
    download_success = download_emnist_zip_manually()
    if not download_success:
        logger.error("Could not obtain EMNIST archive programmatically from mirrors.")
        return
        
    try:
        cache_dir = os.path.expanduser("~/.cache/emnist")
        zip_path = os.path.join(cache_dir, "emnist.zip")
        
        # Use our custom ZIP extractor and IDX parser
        x_train, y_train, x_test, y_test = extract_and_parse_emnist_balanced(zip_path)
        
        np.savez_compressed(filepath, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
        logger.info(f"EMNIST Balanced saved successfully to {filepath}")
        
        # Save EMNIST Balanced character mapping
        # 0-9: digits, 10-35: uppercase, 36-46: lowercase (a, b, d, e, f, g, h, n, q, r, t)
        emnist_map = {
            0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
            10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I',
            19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R',
            28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z',
            36: 'a', 37: 'b', 38: 'd', 39: 'e', 40: 'f', 41: 'g', 42: 'h', 43: 'n', 44: 'q',
            45: 'r', 46: 't'
        }
        save_label_mapping(emnist_map, "dataset/EMNIST/emnist_labels.json")
    except Exception as e:
        logger.error(f"Failed to extract EMNIST: {e}")



def print_manual_instructions():
    """Prints guidance for downloading A-Z and IAM datasets manually."""
    print("=" * 60)
    print("MANUAL DATASET INSTRUCTIONS:")
    print("=" * 60)
    print("1. A-Z Handwritten Alphabets Dataset (Optional)")
    print("   Download 'A_Z Handwritten Data.csv' from Kaggle:")
    print("   https://www.kaggle.com/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format")
    print("   Place the CSV file in: dataset/A_Z/A_Z Handwritten Data.csv")
    print("\n2. IAM Handwriting Database (Optional for Word Recognition)")
    print("   Register and download the word dataset from FKI Bern:")
    print("   https://fki.tic.heia-fr.ch/databases/download-the-iam-handwriting-database")
    print("   Download 'words.tgz' and 'xml.tgz'.")
    print("   Extract them into: dataset/IAM/words/ and dataset/IAM/xml/")
    print("   (Note: If not present, our train_crnn.py script will generate synthetic word images)")
    print("=" * 60)

if __name__ == "__main__":
    create_directories()
    download_mnist()
    download_emnist()
    print_manual_instructions()

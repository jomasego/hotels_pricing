import gdown
import os

def download_file_from_google_drive(file_id, destination):
    """Download a file from Google Drive using its file ID."""
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, destination, quiet=False)

def main():
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Download the pricing matrix
    print("Downloading pricing matrix...")
    download_file_from_google_drive(
        '1FbILDz4rMMMJQXBQLRsQCDA9ULmqLoB4',
        'data/pricing_matrix_30x30.csv'
    )
    
    print("\nDownload completed successfully!")

if __name__ == "__main__":
    main()

# `pip install kaggle` at first
# `pip install --upgrade pip setuptools` to update the dependencies
import kaggle
import os
import zipfile

dataset_slug = 'jockeroika/life-style-data'
expected_csv = 'lifestyle_data.csv'
zip_file = 'life-style-data.zip'


def download_lifestyle_dataset():
    if os.path.exists(expected_csv):
        print(f"'{expected_csv}' has downloaded, skipping...")
        return

    if os.path.exists(zip_file):
        print(f"'{zip_file}' existed，unzipping...")
        unzip_file(zip_file)
        if os.path.exists(expected_csv):
            print(f"Successfully unzip '{expected_csv}'。")
            return
        else:
            print("Fail to unzip..")

    print(f"Downloading form Kaggle -> '{dataset_slug}'...")
    kaggle.api.dataset_download_files(dataset_slug, path='.', unzip=True)
    print(f"Successfully downloaded '{expected_csv}'")


def unzip_file(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('.')
    os.remove(zip_path)


if __name__ == "__main__":
    download_lifestyle_dataset()
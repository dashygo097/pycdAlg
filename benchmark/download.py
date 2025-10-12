import gzip
import os
import shutil
import urllib.request

import requests
from tqdm import tqdm


def download_dataset(url, dir=None, filename=None):
    os.makedirs(dir, exist_ok=True) if dir else None
    if filename is None:
        filename = url.split("/")[-1]
    filename = os.path.join(dir, filename) if dir else filename

    def progress_hook(t):
        last_b = [0]

        def update_to(b=1, bsize=1, tsize=None):
            if tsize is not None:
                t.total = tsize
            t.update((b - last_b[0]) * bsize)
            last_b[0] = b

        return update_to

    with tqdm(
        unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc=filename
    ) as t:
        urllib.request.urlretrieve(url, filename=filename, reporthook=progress_hook(t))

    extracted_filename = filename.replace(".gz", "")
    print(f"Extracting to {extracted_filename}...")

    with gzip.open(filename, "rb") as f_in:
        with open(extracted_filename, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    print(f"Extraction complete! File saved as {extracted_filename}")

    return extracted_filename


if __name__ == "__main__":
    dataset_url = "https://snap.stanford.edu/data/web-NotreDame.txt.gz"

    try:
        extracted_file = download_dataset(
            dataset_url,
            dir="./datasets/webND",
            filename="web-NotreDame.txt.gz",
        )

        compressed_size = os.path.getsize("./datasets/webND/web-NotreDame.txt.gz")
        extracted_size = os.path.getsize(extracted_file)

        print("\nDownload Summary:")
        print(f"Compressed file size: {compressed_size:,} bytes")
        print(f"Extracted file size: {extracted_size:,} bytes")
        print(f"Compression ratio: {extracted_size / compressed_size:.2f}x")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading the file: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

import urllib.request
import json
import pandas as pd

ROLL = 2025201036

def fetch_dataset():

    data = []
    index = 0

    print("Starting dataset download...")

    while True:

        url = f"http://preon.iiit.ac.in:8026/api/data?roll={ROLL}&index={index}"

        try:
            with urllib.request.urlopen(url) as response:

                obj = json.loads(response.read().decode())

                features = obj["features"]
                label = obj["label"]

                data.append(features + [label])

                if index % 100 == 0:
                    print("Downloaded samples:", index)

                index += 1

        except urllib.error.HTTPError as e:

            if e.code == 404:
                print("Finished downloading dataset")
                break

    df = pd.DataFrame(data)

    print("Dataset size:", df.shape)

    return df

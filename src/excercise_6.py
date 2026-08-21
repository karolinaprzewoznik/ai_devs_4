import pandas as pd
import requests
import io
import os

API_KEY = os.environ.get("AIDEVS_API_KEY")
BASE_URL = "https://hub.ag3nts.org"


def process_categorization():
    """
    This function processes the categorization of data by sending requests to the API.
    It first resets the categorization task, then retrieves a CSV file containing data
    to be categorized.

    For each row in the CSV, it sends a request to categorize the description based on
    the specified criteria.
    """

    # Reset the categorization task before processing new data.
    requests.post(
        f"{BASE_URL}/verify",
        json={"apikey": API_KEY, "task": "categorize", "answer": {"prompt": "reset"}},
    )

    # Download the CSV file containing items to be categorized.
    csv_url = f"{BASE_URL}/data/{API_KEY}/categorize.csv"
    response = requests.get(csv_url)

    # Load the CSV data into a DataFrame for iteration.
    df = pd.read_csv(io.StringIO(response.text))

    # Prompt template used to classify each item as dangerous or neutral.
    prompt_template = """ID {code}: {description}. If this mentions reactor, fuel, core, or cassette, reply NEU. Else if dangerous, DNG. Else NEU. Output ONLY DNG or NEU."""

    # Send one categorization request per row in the dataset.
    for _, row in df.iterrows():
        code = row["code"]
        description = row["description"]

        # Build the API payload for the current row.
        payload = {
            "apikey": API_KEY,
            "task": "categorize",
            "answer": {
                "prompt": prompt_template.format(code=code, description=description)
            },
        }

        # Submit the classification request and print the result.
        res = requests.post(f"{BASE_URL}/verify", json=payload)
        print(f"Code {code} | Odpowiedź: {res.json()}")


if __name__ == "__main__":
    process_categorization()

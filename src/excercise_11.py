import os
import json
import time
import zipfile
import requests
from pathlib import Path
from google import genai

AIDEVS_API_KEY = os.environ.get("AIDEVS_API_KEY")
VERIFY_URL = "https://hub.ag3nts.org/verify"
SENSORS_ZIP_URL = "https://hub.ag3nts.org/dane/sensors.zip"
DATA_DIR = Path("sensors_data")
CACHE_FILE = Path("llm_cache.json")

RANGES = {
    "temperature_K": (553, 873),
    "pressure_bar": (60, 160),
    "water_level_meters": (5.0, 15.0),
    "voltage_supply_v": (229.0, 231.0),
    "humidity_percent": (40.0, 80.0),
}

SENSOR_TOKEN_MAP = {
    "temperature": "temperature_K",
    "pressure": "pressure_bar",
    "water": "water_level_meters",
    "voltage": "voltage_supply_v",
    "humidity": "humidity_percent",
}


def download_and_extract_sensors() -> None:
    """
    Downloads the sensors.zip archive from the specified URL and extracts
    its contents into the DATA_DIR. If the DATA_DIR already exists and
    contains JSON files, it assumes the data is already present and does
    not download again.
    """

    if DATA_DIR.exists() and any(DATA_DIR.glob("*.json")):
        print("[*] Sensor data is already present locally.")
        return

    print("[*] Downloading sensors.zip archive...")
    response = requests.get(SENSORS_ZIP_URL)
    response.raise_for_status()

    zip_path = Path("sensors.zip")
    zip_path.write_bytes(response.content)

    print("[*] Extracting archive...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR)
    print("[*] Extracted successfully.")


def load_cache() -> dict:
    """
    Loads the cache from the CACHE_FILE if it exists. If the file does not
    exist or is corrupted, it returns an empty dictionary.
    """

    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_cache(cache_data: dict) -> None:
    """
    Saves the provided cache_data dictionary to the CACHE_FILE in JSON format.
    """

    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, ensure_ascii=False, indent=2)


def analyze_notes_batch_with_llm(
    notes_batch: list[str], client: genai.Client
) -> dict[str, bool]:
    """
    Analyzes a batch of operator notes using the LLM to determine if each note
    indicates an anomaly (True) or not (False). Returns a dictionary mapping
    the original notes to their boolean classification.
    """

    # Create a mapping of indices to notes for easier reference in the prompt
    indexed_batch = {str(i): note for i, note in enumerate(notes_batch)}

    prompt = (
        "Przeanalizuj poniższy słownik notatek operatorów elektrowni (klucz to"
        " indeks, wartość to treść notatki).\nZasada klasyfikacji:\n- Jeśli"
        " notatka wskazuje, że wszystko jest stabilne, w porządku, nie wymaga"
        " interwencji, oceń ją jako FALSE (brak błędu).\n- Jeśli operator zgłasza"
        " anomalię, ostrzeżenie, błąd, problem, odchylenie lub konieczność"
        " interwencji (biorąc pod uwagę kontekst negacji, np. 'no issue' to"
        " FALSE), oceń ją jako TRUE.\n\nNotatki do"
        f" oceny:\n{json.dumps(indexed_batch, indent=2)}\n\nZwróć WYŁĄCZNIE"
        " poprawny obiekt JSON, w którym kluczami są DOKŁADNIE te same indeksy"
        " (jako stringi, np. '0', '1'), a wartościami są boolean (true/false)."
        " Żadnego dodatkowego tekstu."
    )

    while True:
        try:
            response = client.models.generate_content(
                model="gemini-3.6-flash", contents=prompt
            )
            text = response.text.replace("```json", "").replace("```", "").strip()
            raw_result = json.loads(text)

            # Convert the raw result back to the original notes using the indexed_batch mapping
            resolved_result = {}
            for str_idx, is_anomaly in raw_result.items():
                if str_idx in indexed_batch:
                    resolved_result[indexed_batch[str_idx]] = bool(is_anomaly)
            return resolved_result
        except Exception:
            print(
                "[!] Error communicating with LLM or parsing. Retrying in 10 seconds..."
            )
            time.sleep(10)


def process_sensors() -> list:
    """
    Main function to process sensor data. It downloads and extracts sensor data,
    validates the readings, analyzes operator notes using the LLM, and returns
    a list of anomalies.
    """

    download_and_extract_sensors()
    client = genai.Client()
    cache = load_cache()

    json_files = sorted(DATA_DIR.glob("*.json"))
    print(f"[*] Found {len(json_files)} JSON files for analysis.")

    anomalies = []
    valid_files_to_check = {}
    unique_notes = set()

    invalid_sensors_count = 0

    for file_path in json_files:
        file_id = file_path.stem
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            anomalies.append(file_id)
            continue

        sensor_type_str = data.get("sensor_type", "")
        active_tokens = [t.strip() for t in sensor_type_str.split("/")]
        active_fields = {
            SENSOR_TOKEN_MAP[token]
            for token in active_tokens
            if token in SENSOR_TOKEN_MAP
        }

        is_data_valid = True
        for field_name, (min_val, max_val) in RANGES.items():
            val = data.get(field_name, 0)
            if field_name in active_fields:
                if not (min_val <= val <= max_val):
                    is_data_valid = False
            else:
                if val != 0:
                    is_data_valid = False

        operator_note = data.get("operator_notes", "")

        if not is_data_valid:
            anomalies.append(file_id)
            invalid_sensors_count += 1
        else:
            valid_files_to_check[file_id] = operator_note
            unique_notes.add(operator_note)

    notes_to_fetch = [note for note in unique_notes if note not in cache]
    print(
        f"[+] Physical preprocessing:\n"
        f" - Readings outside the norm (physical anomalies):"
        f" {invalid_sensors_count}\n"
        f" - Unique notes to verify by LLM:"
        f" {len(unique_notes)}\n"
        f" - Already in cache: {len(unique_notes) - len(notes_to_fetch)}\n"
        f" - To download by LLM now: {len(notes_to_fetch)}"
    )

    batch_size = 200
    for i in range(0, len(notes_to_fetch), batch_size):
        batch = notes_to_fetch[i : i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(notes_to_fetch) + batch_size - 1) // batch_size

        print(
            f"[*] Analyzing note batch with LLM {batch_num}/{total_batches}"
            f" (size: {len(batch)})..."
        )
        batch_analysis = analyze_notes_batch_with_llm(batch, client)
        cache.update(batch_analysis)
        save_cache(cache)

    for file_id, note in valid_files_to_check.items():
        if cache.get(note, False):
            anomalies.append(file_id)

    print(f"[+] Total number of detected anomalies: {len(anomalies)}")
    return sorted(anomalies, key=lambda x: int(x))


def send_answer(anomalies: list) -> None:
    """
    Sends the list of detected anomalies to the verification server.
    """

    payload = {
        "apikey": AIDEVS_API_KEY,
        "task": "evaluation",
        "answer": {"recheck": anomalies},
    }

    print(f"[*] Sending answer to {VERIFY_URL} (number of IDs: {len(anomalies)})...")
    response = requests.post(VERIFY_URL, json=payload)
    print(f"Status: {response.status_code}")
    try:
        print("Answer:", response.json())
    except Exception:
        print("Text answer:", response.text)


if __name__ == "__main__":
    detected_anomalies = process_sensors()
    send_answer(detected_anomalies)

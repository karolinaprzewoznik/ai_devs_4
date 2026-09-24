import csv
import os
import re
import json
import base64
import mimetypes
import xml.etree.ElementTree as ET
from decimal import Decimal, ROUND_HALF_UP
from collections import defaultdict

import requests
from google import genai
from google.genai import types

API_KEY = os.environ.get("AIDEVS_API_KEY")
VERIFY_URL = "https://hub.ag3nts.org/verify"
TASK = "radiomonitoring"
OUT_DIR = "radio_data"

client = genai.Client()


def call_api(answer: dict):
    """
    Sends a POST request to the API verification endpoint with the specified answer payload.

    Args:
        answer (dict): The dictionary containing the task-specific answer data.

    Returns:
        dict: The parsed JSON response returned by the API server.

    Raises:
        requests.exceptions.HTTPError: If the HTTP request returns an unsuccessful status code.
    """

    payload = {"apikey": API_KEY, "task": TASK, "answer": answer}
    r = requests.post(VERIFY_URL, json=payload, timeout=60)
    r.raise_for_status()

    return r.json()


NOISE_TOKENS = re.compile(
    r"\b(kshh+|ksss+h?|szzz+|bzzz+t?|trzask|pisk|szum|khhh+h?)\b", re.IGNORECASE
)


def is_mostly_noise(text: str) -> bool:
    """
    Determines whether a given text consists predominantly of noise tokens.

    Args:
        text (str): The input text string to analyze.

    Returns:
        bool: True if the text contains no word tokens or if the proportion
              of noise tokens exceeds 15%; otherwise False.
    """

    tokens = re.findall(r"\w+", text)
    if not tokens:
        return True
    noisy = len(NOISE_TOKENS.findall(text))

    return noisy / max(len(tokens), 1) > 0.15


def ext_from_mime(mime: str) -> str:
    """
    Guesses the file extension corresponding to a given MIME type.

    Args:
        mime (str): The MIME type string to evaluate (e.g., 'application/json').

    Returns:
        str: The guessed file extension including the leading dot (e.g., '.json'),
              or '.bin' as a fallback if the MIME type cannot be resolved.
    """

    ext = mimetypes.guess_extension(mime or "")

    return ext or ".bin"


def collect_materials():
    """
    Initiates a radio monitoring session and iteratively collects all captured materials,
    categorizing text transcriptions into clean and noisy data, while decoding and saving
    binary attachments to disk.

    Returns:
        tuple: A 3-tuple containing:
              - transcriptions_clean (list of str): Filtered high-quality text transcripts.
              - transcriptions_noisy (list of str): Transcripts identified as mostly noise.
              - saved_files (list of dict): Metadata and file paths for all successfully
              saved binary files.
    """

    os.makedirs(OUT_DIR, exist_ok=True)

    start_resp = call_api({"action": "start"})
    print("Start:", start_resp)

    transcriptions_clean = []
    transcriptions_noisy = []
    saved_files = []

    idx = 0
    while True:
        resp = call_api({"action": "listen"})

        if resp.get("code") == 101:
            print("End of listening:", resp.get("message"))
            break

        if "transcription" in resp:
            text = resp["transcription"]
            if is_mostly_noise(text):
                transcriptions_noisy.append(text)
            else:
                transcriptions_clean.append(text)
            continue

        if "attachment" in resp:
            idx += 1
            mime = resp.get("meta", "application/octet-stream")
            ext = ext_from_mime(mime)
            raw = base64.b64decode(resp["attachment"])
            path = os.path.join(OUT_DIR, f"attachment_{idx}{ext}")
            with open(path, "wb") as f:
                f.write(raw)
            saved_files.append({"path": path, "mime": mime, "size": len(raw)})
            print(f"{path} ({mime}, {len(raw)}B)")
            continue

        print("Unknown answer, skipping:", resp)

    return transcriptions_clean, transcriptions_noisy, saved_files


def load_json_cities(path):
    """
    Loads and parses JSON data from a specified file path using
    UTF-8 encoding.

    Args:
        path (str): The file path to the JSON file.

    Returns:
        dict or list: The parsed JSON content.
    """

    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_csv_trades(path):
    """
    Loads and parses rows from a CSV file into a list of dictionaries
    using UTF-8 encoding.

    Args:
        path (str): The file path to the CSV file.

    Returns:
        list of dict: A list where each element represents a row from
        the CSV file as a dictionary.
    """

    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    return rows


def is_decoy_xml(path) -> bool:
    """
    Determines whether an XML file is flagged as training or decoy data
    based on its root attributes.

    Args:
        path (str): The file path to the XML file.

    Returns:
        bool: True if the XML root element has a 'trainingData' attribute
        set to "true"; otherwise False.
    """

    try:
        root = ET.parse(path).getroot()
        return root.attrib.get("trainingData", "false").lower() == "true"
    except Exception:
        return False


SYNONYMS = {
    "bydło": "cattle",
    "wołowina": "cattle",
    "krowa": "cattle",
    "kilof": "kilof",
}


def normalize_good(name: str) -> str:
    """
    Normalizes a given item or good name by stripping whitespace and
    converting to lowercase, then maps it to a canonical form using
    a predefined synonyms dictionary if available.

    Args:
        name (str): The raw name of the good or item.

    Returns:
        str: The normalized and mapped canonical name of the good.
    """

    return SYNONYMS.get(name.strip().lower(), name.strip().lower())


def find_syjon_alias(csv_rows, known_city_names):
    """
    Detects which real city name is hidden behind the alias 'Syjon' by comparing
    mirrored trade entries (same action, same trade good pair) between 'Syjon'
    and other cities from the list.

    Args:
        csv_rows (list of dict): The trade records loaded from the CSV file.
        known_city_names (iterable of str): A collection of known real city names.

    Returns:
        tuple: A 3-tuple containing:
              - best_city (str): The name of the city that best matches the
              'Syjon' trade profile.
              - best_score (int): The number of matching trade signature records.
              - scores (defaultdict): A dictionary mapping each city to its
              matching score.
    """

    syjon_rows = [r for r in csv_rows if r["miasto"] == "Syjon"]
    syjon_signature = {
        (r["akcja"], normalize_good(r["towar"]), normalize_good(r["w_zamian"]))
        for r in syjon_rows
    }

    scores = defaultdict(int)
    for city in known_city_names:
        city_rows = [r for r in csv_rows if r["miasto"] == city]
        city_signature = {
            (r["akcja"], normalize_good(r["towar"]), normalize_good(r["w_zamian"]))
            for r in city_rows
        }
        scores[city] = len(syjon_signature & city_signature)

    best_city, best_score = max(scores.items(), key=lambda kv: kv[1])
    if best_score == 0:
        raise RuntimeError("Not possible to match 'Syjon' do any city.")

    return best_city, best_score, scores


def round_area(value: float) -> str:
    """
    Performs true mathematical rounding to two decimal places using standard
    half-up rounding.

    Args:
        value (float or str): The numeric value representing the area to
        be rounded.

    Returns:
        str: The rounded value formatted as a string with exactly two decimal
        places (e.g., '12.34').
    """

    return str(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


# ---------------------------------------------------------------------------
# 3. ANALIZA MODELEM — tylko tam, gdzie kod nie wystarczy (obraz, audio)
# ---------------------------------------------------------------------------


def gemini_extract_text_from_image(image_path: str) -> str:
    """
    Extracts all visible text from an image file using the Gemini model.

    Args:
        image_path (str): The file path to the target image.

    Returns:
        str: The extracted and stripped raw text found in the image.
    """

    with open(image_path, "rb") as f:
        data = f.read()
    mime = mimetypes.guess_type(image_path)[0] or "image/png"
    resp = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=[
            types.Part.from_bytes(data=data, mime_type=mime),
            "Przepisz dokładnie cały tekst widoczny na tym obrazie (np. karteczka, notatka). "
            "Zwróć tylko surowy tekst, bez komentarzy.",
        ],
    )

    return resp.text.strip()


def gemini_transcribe_audio(audio_path: str) -> str:
    """
    Transcribes an audio file into text using the Gemini model.

    Args:
        audio_path (str): The file path to the target audio file.

    Returns:
        str: The raw and stripped transcript text.
    """

    with open(audio_path, "rb") as f:
        data = f.read()
    mime = mimetypes.guess_type(audio_path)[0] or "audio/mpeg"
    resp = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=[
            types.Part.from_bytes(data=data, mime_type=mime),
            "Zrób pełną transkrypcję tego nagrania audio (język polski). "
            "Zwróć tylko surowy tekst transkrypcji.",
        ],
    )

    return resp.text.strip()


def is_probably_decoy_image(image_path: str) -> bool:
    """
    Evaluates whether an image file is likely a decoy or irrelevant using
    lightweight local heuristics (such as file size or basic metadata) before
    sending it to an LLM for analysis.
    Currently defaults to False, deferring the filtering decision to subsequent
    model-based content analysis.

    Args:
        image_path (str): The file path to the target image.

    Returns:
        bool: True if the image is likely a decoy; otherwise False.
    """

    return False


def extract_phone(text: str):
    """
    Extracts a phone number matching a 9-digit pattern from a text string
    and normalizes it by removing any spaces or separators.

    Args:
        text (str): The input text string to search within.

    Returns:
        str or None: The cleaned phone number containing only digits,
              or None if no matching phone number is found.
    """

    m = re.search(r"\b(\d{3}[-\s]?\d{3}[-\s]?\d{3})\b", text)
    if not m:
        return None

    return re.sub(r"\D", "", m.group(1))


def extract_warehouses_count(all_text: str):
    """
    Extracts the total number of warehouses located in the city of Syjon / Skarszewy
    from the aggregated text using a regex heuristic, falling back to a Gemini model
    query if the regex pattern does not match.

    Args:
        all_text (str): The combined text containing radio transcriptions and documents.

    Returns:
        int or None: The parsed number of warehouses, or None if it cannot be determined.
    """

    m = re.search(
        r"(\d+)\s+magazyn(?:y|ów|ow)?",
        all_text,
        re.IGNORECASE,
    )
    if m:
        return int(m.group(1))

    resp = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=(
            "W poniższym tekście (transkrypcje radiowe + dokumenty) znajdź liczbę "
            "magazynów znajdujących się w mieście Syjon / Skarszewy. "
            "Odpowiedz WYŁĄCZNIE samą liczbą całkowitą, bez żadnego innego tekstu.\n\n"
            + all_text
        ),
    )
    digits = re.sub(r"\D", "", resp.text.strip())

    return int(digits) if digits else None


def main():
    """
    Orchestrates the entire radio monitoring data pipeline: collects materials,
    parses structural files (JSON, CSV, XML), resolves the real name of the city
    behind the 'Syjon' alias, performs OCR on images to find the contact phone number,
    transcribes audio files, extracts the warehouse count from text data, and
    finally submits the compiled final report to the API verification endpoint.
    """

    clean_texts, noisy_texts, files = collect_materials()

    print(
        f"\nPure transcriptions: {len(clean_texts)}, noise: {len(noisy_texts)}, files: {len(files)}"
    )

    # --- Local analysis of structural files ---
    json_path = next(f["path"] for f in files if f["mime"] == "application/json")
    csv_path = next(f["path"] for f in files if f["mime"] == "text/csv")
    xml_path = next((f["path"] for f in files if f["mime"] == "text/xml"), None)

    cities = load_json_cities(json_path)
    city_names = [c["name"] for c in cities]
    trades = load_csv_trades(csv_path)

    if xml_path and is_decoy_xml(xml_path):
        print(f"[FILTER] {xml_path} signed as training data - skipping.")

    # --- Establishing the real name of city "Syjon" ---
    real_name, score, all_scores = find_syjon_alias(trades, city_names)
    print(f"Syjon = {real_name} (match: {score} of fitting transactions)")
    print("All results matched:", dict(all_scores))

    city_info = next(c for c in cities if c["name"] == real_name)
    city_area = round_area(city_info["occupiedArea"])
    print(f"cityArea = {city_area}")

    # --- OCR of images ---
    image_files = [f for f in files if f["mime"] in ("image/jpeg", "image/png")]
    phone_number = None
    for img in image_files:
        text = gemini_extract_text_from_image(img["path"])
        print(f"[OCR {img['path']}]: {text}")
        phone = extract_phone(text)
        if phone and re.search(r"syjon|wołowin", text, re.IGNORECASE):
            phone_number = phone

    if not phone_number:
        raise RuntimeError("Not possible to find phone number on any image.")
    print(f"phoneNumber = {phone_number}")

    # --- Audio transcription ---
    audio_files = [f for f in files if f["mime"] == "audio/mpeg"]
    audio_transcripts = [gemini_transcribe_audio(a["path"]) for a in audio_files]
    for t in audio_transcripts:
        print("[AUDIO TRANSCRIPTION]:", t)

    # --- Establishing amount of warehouses ---
    combined_text = "\n".join(clean_texts + audio_transcripts)
    warehouses_count = extract_warehouses_count(combined_text)
    if warehouses_count is None:
        raise RuntimeError(
            "Not possible to establish amount of warehouses automatically - please set up manually."
        )
    print(f"warehousesCount = {warehouses_count}")

    # --- Final report ---
    report = {
        "action": "transmit",
        "cityName": real_name,
        "cityArea": city_area,
        "warehousesCount": warehouses_count,
        "phoneNumber": phone_number,
    }
    print("\nSending report:", report)
    result = call_api(report)
    print("RESULT:", json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

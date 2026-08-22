import requests
import re
import os
from google import genai

API_KEY = os.environ.get("AIDEVS_API_KEY")
BASE_URL = "https://hub.ag3nts.org"

client = genai.Client()


def get_raw_logs():
    """
    Download the raw logs from the Hub. This is a simulation of fetching
    logs from a remote server.

    Returns:
        list: A list of log lines.
    """
    res = requests.get(f"{BASE_URL}/data/{API_KEY}/failure.log")
    return res.text.splitlines()


def smart_compress_logs(raw_logs, hint=""):
    """
    Compress the raw logs using a language model, while preserving the most
    important events.

    Parameters:
        raw_logs (list): The raw log lines.
        hint (str): Optional hint to guide the model in selecting relevant logs.

    Returns:
        str: The compressed log lines as a single string.
    """

    # Prefilter the logs to remove noise and keep only critical, error, and
    # warning levels
    prefiltered = [
        line
        for line in raw_logs
        if any(lvl in line for lvl in ["CRIT", "ERROR", "WARN"])
    ]
    raw_text_chunk = "\n".join(prefiltered)

    print(f"-> Raw logs after initial noise cleaning: {len(prefiltered)} lines.")

    prompt = f"""
    Jesteś analitykiem systemowym. Masz surowe logi z awarii elektrowni.
    Twoim zadaniem jest wyselekcjonowanie i skondensowanie najważniejszych zdarzeń
    prowadzących do awarii.
    
    Wymagania:
    - Zachowaj ŚCISŁY porządek chronologiczny (od najstarszych do najmłodszych).
    - Format każdej linii: [YYYY-MM-DD HH:MM:SS] [POZIOM] KOMPONENT: Opis
    - Wybierz logi kluczowe dla zrozumienia pełnego przebiegu awarii (od pierwszych
    ostrzeżeń po końcowy trip).
    - Całość musi zmieścić się w maksymalnie 45 linijkach (rygorystyczny limit
    tokenów).
    - Zwróć WYŁĄCZNIE gotowe linie logu (jedna pod drugą), bez żadnego dodatkowego
    tekstu, wstępów czy bloków markdown.
    {f"DODATKOWA WSKAZÓWKA Z CENTRALI (uwzględnij to koniecznie): \
    {hint}" if hint else ""}
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash", contents=[prompt, raw_text_chunk]
    )

    return response.text.strip()


# Main execution loop
raw_logs = get_raw_logs()
current_hint = ""
compressed_hint = ""

for attempt in range(1, 6):
    print(f"\n--- Attempt {attempt} ---")

    compressed_hint = " ".join([compressed_hint, current_hint])

    # 1. Logs compression
    compressed_logs = smart_compress_logs(raw_logs, compressed_hint)
    line_count = len(compressed_logs.splitlines())
    print(f"Generted {line_count} lines.")

    # 2. Sending compressed logs to the central server for verification
    print("Sending to Hub...")
    response = requests.post(
        f"{BASE_URL}/verify",
        json={
            "apikey": API_KEY,
            "task": "failure",
            "answer": {"logs": compressed_logs},
        },
    )
    result = response.json()
    print("Hub response:", result)

    # 3. Check if the response contains the flag or indicates success
    if "FLG" in str(result) or result.get("code") == 1:
        print("🎉 SUCCES! Flag received!")
        break

    # 4. If there is something missing, it is dynamically returned from message
    message = result.get("message", "")
    match = re.search(r"to (?:device )?([A-Z0-9]+)\.", message)
    if match:
        missing_device = match.group(1)
        print(
            f"Raised missing information about: {missing_device}. Sending it as a hint to next attempt."
        )
        current_hint = (
            f"Zadbaj o pełną historię i zdarzenia dla komponentu: {missing_device}"
        )
    else:
        if "token" in message.lower():
            print("Limit of tokens exceeded - model needs to generate less lines.")
            current_hint = (
                "Wygeneruj mniej linii (5 mniej niż poprzednio), skróć opisy."
            )
        else:
            print("Different warning, stopping code execution.")
            break

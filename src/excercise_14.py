import os
import csv
import io
import time
import threading
import requests
from flask import Flask, request, jsonify
from pyngrok import ngrok
from google import genai

app = Flask(__name__)

CSV_BASE_URL = "https://hub.ag3nts.org/dane/s03e04_csv/"
VERIFY_URL = "https://hub.ag3nts.org/verify"
AIDEVS_API_KEY = os.environ.get("AIDEVS_API_KEY", "TWÓJ_KLUCZ_API")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "TWÓJ_KLUCZ_GEMINI")

ai_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


def fetch_csv_data(filename: str) -> list:
    """
    Downloads CSV data from a given URL and returns it as a list of dictionaries.

    Args:
        filename (str): The name of the CSV file to download.

    Returns:
        list: A list of dictionaries representing the CSV rows, or an empty list
        if an error occurs.
    """

    url = f"{CSV_BASE_URL}{filename}"

    try:
        response = requests.get(url)
        if response.status_code == 200:
            f = io.StringIO(response.text)
            reader = csv.DictReader(f)
            return list(reader)

    except Exception as e:
        print(f"[Error] Error fetching {filename}: {e}")

    return []


print("[Info] Loading CSV data...")
cities_data = fetch_csv_data("cities.csv")
connections_data = fetch_csv_data("connections.csv")
items_data = fetch_csv_data("items.csv")

city_code_to_name = {row["code"]: row["name"] for row in cities_data}
item_code_to_name = {row["code"]: row["name"] for row in items_data}

item_to_cities = {}
for conn in connections_data:
    c_name = city_code_to_name.get(conn["cityCode"])
    i_name = item_code_to_name.get(conn["itemCode"])
    if c_name and i_name:
        item_to_cities.setdefault(i_name, []).append(c_name)

print(f"[Info] Loaded {len(cities_data)} cities and {len(items_data)} items.")


@app.route("/api/search_knowledge", methods=["POST"])
def search_knowledge_tool():
    """
    Endpoint API to search knowledge based on the provided parameters. It processes the
    request, filters candidates based on the query, and uses the Gemini API to generate
    a response.

    Returns:
        Response: A JSON response containing the output of the search.
    """

    data = request.json or {}
    raw_params = data.get("params", "")

    if isinstance(raw_params, dict):
        params = " ".join([str(v) for v in raw_params.values()]).strip()
    else:
        params = str(raw_params).strip()

    print(f"\n[Agent's Query]: {params}")

    if not params:
        return jsonify({"output": "Error: No query parameters provided."})

    response_text = ""
    params_lower = params.lower()

    if ai_client:
        candidates = []
        is_turbine = "turbina" in params_lower or "wiatrowa" in params_lower
        is_battery = "akumulator" in params_lower
        is_inverter = "inwerter" in params_lower

        for item, cities in item_to_cities.items():
            item_lower = item.lower()
            match = False
            if is_turbine and "turbina" in item_lower:
                match = True
            elif is_battery and "akumulator" in item_lower:
                match = True
            elif is_inverter and "inwerter" in item_lower:
                match = True
            elif not is_turbine and not is_battery and not is_inverter:
                match = True

            if match:
                if "48v" in params_lower:
                    if "48v" in item_lower:
                        for city in cities:
                            candidates.append(f"Przedmiot: {item} | Miasto: {city}")
                else:
                    for city in cities:
                        candidates.append(f"Przedmiot: {item} | Miasto: {city}")

        if not candidates:
            for item, cities in item_to_cities.items():
                item_lower = item.lower()
                if (
                    (is_turbine and "turbina" in item_lower)
                    or (is_battery and "akumulator" in item_lower)
                    or (is_inverter and "inwerter" in item_lower)
                ):
                    for city in cities:
                        candidates.append(f"Przedmiot: {item} | Miasto: {city}")

        prompt = f"""
        Jesteś precyzyjnym systemem dopasowującym bazy danych dla agenta negocjacyjnego.
        Zapytanie agenta: "{params}"

        Wybierz z poniższej przefiltrowanej listy kandydatów te pozycje, które najbardziej
        precyzyjnie odpowiadają zapytaniu (zwróć szczególną uwagę na dopasowanie napięcia
        np. 48V oraz typu urządzenia).
        Zwróć wynik DOKŁADNIE jako maksymalnie 3 linie w formacie:
        Przedmiot: [dokładna nazwa przedmiotu] -> Miasto: [nazwa miasta]
        Żadnego dodatkowego tekstu ani wyjaśnień.

        Kandydaci:
        """ + "\n".join(candidates[:100])

        try:
            response = ai_client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt,
            )
            if response and response.text:
                gemini_output = response.text.strip()
                print(f"[Gemini Output]:\n{gemini_output}")

                valid_lines = [
                    line.strip()
                    for line in gemini_output.split("\n")
                    if "Przedmiot:" in line and "-> Miasto:" in line
                ]
                if valid_lines:
                    response_text = "Results: " + " | ".join(valid_lines[:3])
        except Exception as e:
            print(f"[Error Gemini API]: {e}")

    if not response_text:
        response_text = "Results: No matches found."

    if len(response_text.encode("utf-8")) > 490:
        response_text = response_text[:450] + "..."

    print(f"[Tool Output]: {response_text}")

    return jsonify({"output": response_text})


def run_flask_app():
    """
    Runs the Flask application on the specified host and port.
    """

    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)


def register_and_verify(public_url: str):
    """
    Registers the tool with the central server and verifies its functionality.

    Args:
        public_url (str): The public URL of the tool to be registered.
    """

    tool_url = f"{public_url}/api/search_knowledge"
    payload = {
        "apikey": AIDEVS_API_KEY,
        "task": "negotiations",
        "answer": {
            "tools": [
                {
                    "URL": tool_url,
                    "description": "Narzędzie oparte o Gemini do precyzyjnego wyszukiwania komponentów i miast.",
                }
            ]
        },
    }

    print("[Info] Registering the tool in Hub...")
    requests.post(VERIFY_URL, json=payload)
    print("[Info] Waiting 45 seconds...")
    time.sleep(45)

    check_payload = {
        "apikey": AIDEVS_API_KEY,
        "task": "negotiations",
        "answer": {"action": "check"},
    }
    res = requests.post(VERIFY_URL, json=check_payload)
    print(f"HTTP Status: {res.status_code}\nHub Response: {res.text}")


if __name__ == "__main__":
    threading.Thread(target=run_flask_app, daemon=True).start()
    public_url = ngrok.connect(5000).public_url
    print(f"\n[INFO] Public Ngrok URL: {public_url}/api/search_knowledge\n")

    time.sleep(2)
    register_and_verify(public_url)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping...")

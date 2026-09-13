import os
import json
import time
import requests
from google import genai
from google.genai import types

AIDEVS_API_KEY = os.environ.get("AIDEVS_API_KEY")
VERIFY_URL = "https://hub.ag3nts.org/verify"
DRONE_DOCS_URL = "https://hub.ag3nts.org/dane/drone.html"
MAP_URL = f"https://hub.ag3nts.org/data/{AIDEVS_API_KEY}/drone.png"

client = genai.Client()


def download_map():
    """
    Downloads the drone map from the specified URL and saves it as 'drone_map.png'.
    If the download is successful, it prints a success message. If there is an error,
    it prints an error message with the HTTP status code.
    """

    print("[*] Downloading drone map...")
    response = requests.get(MAP_URL)
    if response.status_code == 200:
        with open("drone_map.png", "wb") as f:
            f.write(response.content)
        print("[+] Map saved as drone_map.png")
    else:
        print(f"[!] Error downloading map: {response.status_code}")


def get_documentation():
    """
    Fetches the drone documentation from the specified URL and returns it as text.
    """

    print("[*] Downloading drone documentation...")
    response = requests.get(DRONE_DOCS_URL)
    return response.text


def find_dam_coordinates():
    """
    Analyzes the drone map image using Gemini Vision to find the coordinates of
    the dam. It reads the image file 'drone_map.png', sends it to the Gemini
    Vision model for analysis, and expects a JSON response with the coordinates
    of the dam in the format {"x": <column>, "y": <row>}. If the response cannot
    be parsed as JSON, it returns default coordinates (2, 3).
    """

    print("[*] Analyzing drone map with Gemini Vision...")
    with open("drone_map.png", "rb") as f:
        image_bytes = f.read()

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            (
                "To jest mapa siatkowa elektrowni. Przy tamie celowo wzmocniono"
                " intensywność koloru wody. Podaj współrzędne x (kolumna) i y"
                " (wiersz) sektora, w którym znajduje się tama. Odpowiedz wyłącznie"
                ' w formacie JSON, np. {"x": 3, "y": 4}'
            ),
        ],
    )
    print(f"[+] Gemini Vision response: {response.text}")

    # Try to parse the response as JSON and extract the coordinates
    try:
        # Clean the response text to ensure it's valid JSON
        clean_text = response.text.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_text)
        return data.get("x"), data.get("y")
    except Exception:
        # Default coordinates if parsing fails
        return 2, 3


def send_to_api(instructions):
    """
    Sends the given instructions to the AIDEVS API for verification. It constructs
    a payload with the API key, task type, and instructions, then makes a POST
    request to the VERIFY_URL. It prints the status code of the response and
    attempts to return the JSON response. If parsing fails, it returns the raw text.
    """

    payload = {
        "apikey": AIDEVS_API_KEY,
        "task": "drone",
        "answer": {"instructions": instructions},
    }
    print(f"\n[*] Sending instructions to API: {instructions}")
    response = requests.post(VERIFY_URL, json=payload)
    print(f"Status: {response.status_code}")

    try:
        return response.json()
    except Exception:
        return {"raw_text": response.text}


def run_agent():
    """
    Main function to run the agent. It downloads the map and documentation,
    finds the dam coordinates, and enters a loop where it interacts with the LLM
    to generate instructions for the drone. It sends these instructions to the
    API and checks for success or errors, providing feedback to the agent as needed.
    """

    download_map()
    docs = get_documentation()
    x, y = find_dam_coordinates()
    print(f"[+] Found dam coordinates: x={x}, y={y}")

    # Początkowy prompt dla agenta LLM, który będzie sterował pętlą
    system_prompt = f"""
    Jesteś agentem sterującym dronem bojowym DRN-BMB7 w grze AIdevs.
    Twoim zadaniem jest oszukać system bezpieczeństwa: cel misji ma być
    formalnie ustawiony na elektrownię (PWR6132PL), ale fizycznie bomba
    ma spaść na tamę o współrzędnych x={x}, y={y} na mapie.
    
    Dokumentacja API drona:
    {docs}
    
    Zwracasz WYŁĄCZNIE tablicę JSON z listą instrukcji (pole 'instructions'),
    bez żadnego dodatkowego tekstu markdown, żebym mógł bezpośrednio przekazać
    je do API.
    
    Przykład formatu odpowiedzi:
    ["selfCheck", "set(engineON)", "set(100%)", "set(10m)", "setDestinationObject(
    PWR6132PL)", "set({x},{y})", "set(destroy)", "flyToLocation"]
    Musisz dobrać odpowiednie komendy z dokumentacji, aby osiągnąć cel (zniszczenie
    tamy / dostarczenie wody), nie niszcząc samej elektrowni.
    """

    # Agent LLM chat session
    chat = client.chats.create(model="gemini-3.6-flash")

    # First attempt to generate instructions
    print("[*] Attempting to generate instructions - first attempt...")
    response = chat.send_message(system_prompt)

    for attempt in range(1, 6):
        raw_response = response.text.strip()
        print(f"\n--- Attempt {attempt} ---")
        print(f"Suggested by agent:\n{raw_response}")

        try:
            # Attempt to extract JSON from the agent's response
            if "```json" in raw_response:
                json_str = raw_response.split("```json")[1].split("```")[0].strip()
            elif "```" in raw_response:
                json_str = raw_response.split("```")[1].split("```")[0].strip()
            else:
                json_str = raw_response

            instructions = json.loads(json_str)
            if isinstance(instructions, dict) and "instructions" in instructions:
                instructions = instructions["instructions"]

        except Exception as e:
            print(f"[!] Error parsing agent's response to JSON: {e}")
            # If parsing fails, ask the agent to return a clean JSON array
            response = chat.send_message(
                "That is not a valid JSON array of instructions. Return ONLY a clean JSON array"
                ' of instructions like ["command1", "command2"].'
            )
            continue

        # Call the API with the instructions
        api_result = send_to_api(instructions)
        print(f"API response: {api_result}")

        # Validate the API response and check for success
        result_str = json.dumps(api_result)
        if "FLG:" in result_str:
            print(f"\n[SUCCESS!] Found flag: {result_str}")
            break

        # If the API response indicates an error, provide feedback to the agent
        feedback = (
            f"API response: {api_result}. Correct the instructions and"
            " provide a new JSON array."
        )
        response = chat.send_message(feedback)
        time.sleep(2)


if __name__ == "__main__":
    run_agent()

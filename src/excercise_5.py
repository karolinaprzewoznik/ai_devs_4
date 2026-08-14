import json
import os
import requests
import time

from google import genai

# Configuration
API_KEY = os.environ.get("AIDEVS_API_KEY")
VERIFY_URL = "https://hub.ag3nts.org/verify"
TASK_NAME = "railway"

client = genai.Client()


def call_api(action_payload):
    """
    Sends a request to the API and handles 503 errors and dynamic rate_limit limits
    (retry_after).

    Parameters:
    - action_payload (dict): The payload to send to the API.

    Returns:
    - dict: The JSON response from the API.
    """
    payload = {"apikey": API_KEY, "task": TASK_NAME, "answer": action_payload}
    while True:
        try:
            resp = requests.post(VERIFY_URL, json=payload, timeout=30)

            # Handle network/server-side 503 errors
            if resp.status_code == 503:
                print("[API] Received 503 (simulated overload). Waiting 5 seconds...")
                time.sleep(5)
                continue

            data = resp.json()

            # Check whether the API returned a rate limit with retry_after
            if data.get("code") == -985 and "retry_after" in data:
                wait_time = int(data["retry_after"]) + 1  # Add a safety second
                print(
                    f"[Rate Limit] Limit exceeded. Waiting exactly {wait_time}"
                    " seconds as recommended by the server..."
                )
                time.sleep(wait_time)
                continue  # Retry the same request after waiting

            return data

        except Exception as e:
            print(f"[Network] Error: {e}. Retrying in 3 seconds...")
            time.sleep(3)


def run_railway_agent():
    """
    This function runs the railway agent, which interacts with the API to
    activate the railway route "X-01". It uses a loop to send actions to
    the API and receives responses, which are then analyzed to determine
    the next action. The loop continues until the agent successfully activates
    the route or reaches a maximum number of steps.

    The agent uses the Gemini model to decide on the next action based on
    the history of actions and responses. The function prints the progress
    and any errors encountered during the process.

    Returns:
    None
    """
    print("--- START RAILWAY AGENT ---")

    # 1. Fetch the initial documentation (help)
    current_action = {"action": "help"}
    history = []

    # Agent decision loop (a few steps max)
    for step_num in range(1, 10):
        print(f"\n--- STEP {step_num} ---")
        print(f"Sending to API: {current_action}")

        api_response = call_api(current_action)
        print(f"API response: {api_response}")

        # Check whether the response contains a flag
        response_str = str(api_response)
        if "{FLG:" in response_str:
            print(f"\nSUCCESS! Found a response with the flag: {api_response}")
            break

        # 2. Ask Gemini what to do based on the history and the latest API response
        prompt = f"""
        Jesteś agentem zarządzającym siecią kolejową. Twoim celem jest aktywacja (otwarcie)
        trasy o nazwie "X-01".
        
        Oto historia ostatnich działań i odpowiedzi z API:
        Historia: {history}
        Ostatnia odpowiedź API: {api_response}
        
        Zasady:
        - Przeanalizuj ostatnią odpowiedź API. Jeśli to był 'help', zobacz jakie akcje są
        dostępne i jakie parametry wymagają.
        - Zdecyduj, jaką akcję (JSON) wykonać jako następną, aby przybliżyć się do celu
        (aktywacja trasy X-01).
        - Zwróć WYŁĄCZNIE poprawny obiekt JSON z kluczem "action" oraz wymaganymi parametrami
        (np. {{"action": "...", "route": "...", ...}}). Nie dodawaj żadnego tekstu ani bloków
        markdown ```.
        """

        llm_response = client.models.generate_content(
            model="gemini-2.5-flash", contents=prompt
        )

        # Clean the model response of markdown markers, if present
        next_action_str = (
            llm_response.text.strip().replace("```json", "").replace("```", "").strip()
        )

        try:
            current_action = json.loads(next_action_str)
        except json.JSONDecodeError:
            print(
                f"LLM decision parsing error: {next_action_str}. Retrying with the help request."
            )
            current_action = {"action": "help"}

        history.append({"sent": current_action, "received": api_response})
        time.sleep(2)  # Short pause to avoid rate limits


run_railway_agent()

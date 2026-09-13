import os
import json
import time
import requests
from google import genai

AIDEVS_API_KEY = os.environ.get("AIDEVS_API_KEY")
SHELL_URL = "https://hub.ag3nts.org/api/shell"
VERIFY_URL = "https://hub.ag3nts.org/verify"

client = genai.Client()


def run_shell_cmd(command: str) -> dict:
    """
    Sends a shell command to the remote shell API and returns the JSON response.
    If the response indicates a ban (code -735), it waits for the specified
    number of seconds before retrying the command.
    """

    payload = {"apikey": AIDEVS_API_KEY, "cmd": command}
    try:
        response = requests.post(SHELL_URL, json=payload)
        res_json = response.json()
    except Exception as e:
        return {"error": str(e)}

    if res_json.get("code") == -735:
        ban_info = res_json.get("ban", {})
        sec = ban_info.get("seconds_left", 10) + 3
        print(f"[System] Ban! Czekam {sec} sekund...")
        time.sleep(sec)
        return run_shell_cmd(command)

    return res_json


def ask_gemini_agent(system_state: str) -> dict:
    """
    Sends the current system state to the Gemini agent and retrieves its decision.
    The agent's response is expected to be a JSON object containing its thought,
    action, and optionally a code to verify if the task is completed.
    """

    prompt = f"""
    Jesteś inteligentnym agentem systemowym Linux rozwiązującym zadanie CTF/firmware.
    Ograniczenia środowiska shella:
    - Dostępne komendy: help, ls [path], cat <path>, cd [path], pwd, rm <file>,
    editline <file><line> <content>, reboot, date, uptime, find <pattern>, history,
    whoami.
    - UWAGA: Komenda 'ls' nie obsługuje flag takich jak `-la`. Używaj np.
    `ls /opt/firmware/cooler`.
    - Plik binarny to `/opt/firmware/cooler/cooler.bin`, a konfiguracja to
    `/opt/firmware/cooler/settings.ini`.
    - Twój cel: skonfigurować poprawnie settings.ini (np. wyłączyć tryb testowy,
    włączyć chłodzenie), użyć właściwego hasła z historii i zdobyć kod ECCS-XXXXX.

    Aktualny stan środowiska i wyniki ostatnich operacji:
    {system_state}

    Zwróć WYŁĄCZNIE poprawny obiekt JSON (bez znaczników markdown typu ```json).

    Wymagany format JSON:
    {{
    "thought": "Krótkie uzasadnienie decyzji",
    "action": "dokładna komenda shellowa LUB FINISH",
    "code_to_verify": null
    }}
    """

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
        config=dict(response_mime_type="application/json"),
    )

    text = response.text.strip()
    return json.loads(text)


def send_verification(code: str):
    """
    Sends the verification code to the central server to confirm task completion.
    """

    payload = {
        "apikey": AIDEVS_API_KEY,
        "task": "firmware",
        "answer": {"confirmation": code},
    }
    response = requests.post(VERIFY_URL, json=payload)
    print(f"HTTP status: {response.status_code}")

    try:
        print("Hub response:", json.dumps(response.json(), indent=2))
    except Exception:
        print("Text response:", response.text)


def main():
    """
    Main function to run the Gemini agent in a loop until the task is completed.
    It maintains a history of actions and system state, sending this information
    to the agent for decision-making.
    """

    print("[*] Continuation of the agent loop with gemini-3.6-flash...")

    # Provide an initial system state to the agent from previous iterations
    history_log = (
        "Poprzednie kroki: Znaleziono ścieżki /opt/firmware/cooler/cooler.bin oraz"
        " /opt/firmware/cooler/settings.ini. Pamiętaj, że ls nie przyjmuje flag"
        " typu -la."
    )

    for step in range(1, 15):
        print(f"\n--- STEP NO {step} ---")

        try:
            decision = ask_gemini_agent(history_log)
        except Exception as e:
            print(f"[Error parsing JSON from model]: {e}")
            continue

        print(f"[Gemini's Thought]: {decision.get('thought')}")
        action = decision.get("action")
        print(f"[Decision]: {action}")

        if action == "FINISH":
            code = decision.get("code_to_verify")
            print(f"[Success] Agent finished the task with code: {code}")
            send_verification(code)
            break

        shell_result = run_shell_cmd(action)
        print(f"[Shell Result]: {json.dumps(shell_result, ensure_ascii=False)}")
        history_log += f"\nCommand: {action} -> Result: {json.dumps(shell_result, ensure_ascii=False)}"


if __name__ == "__main__":
    main()

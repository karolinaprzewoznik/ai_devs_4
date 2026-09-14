import os
import json
import time
import re
import requests
from google import genai

AIDEVS_API_KEY = os.environ.get("AIDEVS_API_KEY")
VERIFY_URL = "https://hub.ag3nts.org/verify"

client = genai.Client()


def send_reactor_cmd(command: str) -> dict:
    """
    Sends a command to the reactor environment and returns the response.

    Args:
        command (str): The command to send to the reactor environment.

    Returns:
        dict: The JSON response from the reactor environment.
    """

    payload = {
        "apikey": AIDEVS_API_KEY,
        "task": "reactor",
        "answer": {"command": command},
    }

    try:
        response = requests.post(VERIFY_URL, json=payload)
        return response.json()

    except Exception as e:
        return {"error": str(e)}


def ask_gemini_reactor_agent(board_context: str) -> dict:
    """
    Asks the Gemini model for the next action based on the current board context.

    Args:
        board_context (str): The current state of the board and robot position.

    Returns:
        dict: The model's decision in JSON format.
    """

    prompt = f"""
    Jesteś inteligentnym agentem sterującym robotem transportowym w zadaniu
    'reactor' (AI_devs). Twoim celem jest przeprowadzenie robota z punktu
    startowego P (dół, lewa) do punktu docelowego G (dół, prawa).

    Zasady gry:
    - Plansza ma 7 kolumn i 5 wierszy.
    - Robot porusza się po ostatnim (5) wierszu za pomocą komend: `start`,
    `reset`, `left`, `right`, `wait`.
    - Nad robotem poruszają się pionowo bloki reaktora `B`. Jeśli blok
    znajdzie się na Twojej pozycji, przegrasz.
    - Każda wysłana komenda porusza światem o 1 krok. Komenda `wait` pozwala
    odczekać ruch bloku bez zmiany pozycji robota.

    Oto aktualny, kontekstowy feedback ze stanu środowiska (plansza i pozycja):
    {board_context}

    Przeanalizuj sytuację i podejmij decyzję o następnym kroku.
    Zwróć WYŁĄCZNIE poprawny obiekt JSON bez znaczników markdown.

    Format JSON:
    {{
    "thought": "Twoja analiza bezpieczeństwa drogi i ruchu bloków",
    "action": "start | right | left | wait"
    }}
    """

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
        config=dict(response_mime_type="application/json"),
    )

    text = response.text.strip()

    return json.loads(text)


def main():
    """
    Main function to run the reactor agent with immediate stop condition.
    The agent will send commands to the reactor environment and process
    the feedback until it reaches the goal or finds a code.

    The loop will run for a maximum of 150 steps, and the agent will make
    decisions based on the current state of the environment.
    """

    print("[*] Starting reactor agent...")

    # Step 0: Initialize the reactor environment
    current_state = send_reactor_cmd("start")
    print(
        "[Initialization start]:",
        json.dumps(current_state, ensure_ascii=False, indent=2),
    )

    # Check for immediate success condition (ECCS code found)
    state_str = json.dumps(current_state, ensure_ascii=False)
    match = re.search(r"ECCS-[A-Za-z0-9]+", state_str)
    if match:
        print(f"\n[SUCCESS] Code found: {match.group(0)}")
        return

    history_context = f"Starting state: {state_str}"

    for step in range(1, 150):
        print(f"\n--- STEP NO {step} ---")

        try:
            decision = ask_gemini_reactor_agent(history_context)
        except Exception as e:
            print(f"[Error when parsing model decision]: {e}")
            time.sleep(1)
            continue

        print(f"[Gemini thought]: {decision.get('thought')}")
        action = decision.get("action")
        print(f"[Decision - Command]: {action}")

        # Execution of the action in the system
        result = send_reactor_cmd(action)
        result_str = json.dumps(result, ensure_ascii=False)
        print(f"[Environment Feedback]: {result_str}")

        # Universal success check (handles ECCS, FLG and reached_goal: true)
        if (
            re.search(r"ECCS-[A-Za-z0-9]+", result_str)
            or re.search(r"FLG:[A-Za-z0-9_]+", result_str)
            or result.get("reached_goal") is True
        ):
            print(
                "\n[!] SUCCESS! Robot has reached the goal and acquired the flag/code. Stopping the loop."
                f" Final response: {result_str}"
            )
            break

        # Automatic failure check (if the robot is hit by a block)
        history_context = f"Previous action: {action} -> Server response: {result_str}"

        time.sleep(0.3)


if __name__ == "__main__":
    main()

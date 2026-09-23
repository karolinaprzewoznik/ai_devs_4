import json
import os
import requests

from google import genai
from google.genai import types

API_KEY = os.environ.get("AIDEVS_API_KEY")
TASK = "domatowo"
URL = "https://hub.ag3nts.org/verify"
MAX_ACTION_POINTS = 300
GEMINI_MODEL = "gemini-3.8-flash"
FOUND_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "found_human": {"type": "BOOLEAN"},
    },
    "required": ["found_human"],
}
CLUSTERS = [
    {
        "name": "gorne",
        "drop_point": "E2",
        "tiles": ["F1", "G1", "F2", "G2"],
    },
    {
        "name": "dolne-lewe",
        "drop_point": "B9",
        "tiles": ["A10", "B10", "C10", "A11", "B11", "C11"],
    },
    {
        "name": "dolne-prawe",
        "drop_point": "H9",
        "tiles": ["H10", "I10", "H11", "I11"],
    },
]


gemini_client = genai.Client()


def call(answer: dict, verbose=True) -> dict:
    payload = {"apikey": API_KEY, "task": TASK, "answer": answer}
    resp = requests.post(URL, json=payload, timeout=30)
    if verbose:
        print(">>>", answer)
        print("<<< status:", resp.status_code)
        print("<<< body:", resp.text)
    resp.raise_for_status()
    data = resp.json()

    if isinstance(data, dict) and "action_points_left" in data:
        points_left = data["action_points_left"]
        print(f"[budget] pozostało {points_left}/{MAX_ACTION_POINTS}")
        if points_left < 10:
            print("[budget] UWAGA: bardzo mało punktów akcji zostało!")

    return data


def reset_board():
    return call({"action": "reset"})


def get_map():
    return call({"action": "getMap"})


def get_logs():
    return call({"action": "getLogs"})


def create_transporter(passengers: int):
    return call({"action": "create", "type": "transporter", "passengers": passengers})


def move(obj_hash: str, where: str):
    return call({"action": "move", "object": obj_hash, "where": where})


def inspect(obj_hash: str):
    return call({"action": "inspect", "object": obj_hash})


def dismount(transporter_hash: str, passengers: int):
    return call(
        {"action": "dismount", "object": transporter_hash, "passengers": passengers}
    )


def call_helicopter(destination: str):
    return call({"action": "callHelicopter", "destination": destination})


def ask_gemini_found(msg: str) -> bool:
    """
    Asks Gemini model if the given message confirms the presence of a living human
    (partisan) in the field, as opposed to old traces, abandoned items, smells,
    animals (mice, rats), or statements denying anyone's presence.

    Parameters:
    - msg (str): The message from the scout's inspection log.

    Returns:
    - bool: True if Gemini confirms the presence of a living human, False otherwise.
    """

    prompt = (
        "Jesteś analitykiem raportów zwiadowczych z operacji ewakuacyjnej. "
        "Dostajesz jeden komunikat zwiadowcy z inspekcji pojedynczego pola. "
        "Oceń, czy ten komunikat POTWIERDZA znalezienie ukrywającego się, "
        "żywego człowieka (partyzanta) na tym polu - w odróżnieniu od "
        "starych śladów, porzuconych przedmiotów, zapachów, zwierząt "
        "(myszy, szczury) czy zdań zaprzeczających obecności kogokolwiek.\n\n"
        f'Komunikat: "{msg}"\n\n'
        "Zwróć wyłącznie JSON zgodny ze schematem."
    )

    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            response_schema=FOUND_SCHEMA,
        ),
    )

    data = json.loads(response.text)
    found = bool(data.get("found_human", False))
    print(f'[gemini] "{msg}" -> found_human={found}')

    return found


def get_last_log_entry():
    """
    Gets the last log entry from the logs returned by get_logs().
    Raises a RuntimeError if the logs list is empty.

    Returns:
    - dict: The last log entry.
    """

    logs = get_logs().get("logs", [])

    if not logs:
        raise RuntimeError("getLogs returned an empty list - something went wrong.")

    return logs[-1]


def run_mission():
    """
    Runs the mission to find and evacuate a hidden partisan using scouts and
    a transporter. The function creates a transporter, moves it to predefined
    clusters, and uses scouts to inspect tiles for the presence of a living human.
    If a human is found, it calls a helicopter to evacuate them.
    Raises a RuntimeError if no human is found in any cluster.

    Returns:
    - dict: The result of the helicopter call if a human is found and evacuated.
    """

    created = create_transporter(passengers=3)

    transporter_hash = created.get("object")
    crew = created.get("crew", [])
    scout_hashes = [member["id"] for member in crew]

    if not transporter_hash or len(scout_hashes) < 3:
        raise RuntimeError(f"There was not possible to create transporter: {created}")

    found_position = None

    for cluster, scout_hash in zip(CLUSTERS, scout_hashes):
        if found_position:
            break

        print(f"=== Cluster: {cluster['name']} ===")

        move(transporter_hash, cluster["drop_point"])
        dismount(transporter_hash, 1)

        for tile in cluster["tiles"]:
            move(scout_hash, tile)
            inspect(scout_hash)

            entry = get_last_log_entry()
            if ask_gemini_found(entry["msg"]):
                found_position = entry["field"]
                print(f">>> Gemini found human at position {found_position}!")
                break

    if not found_position:
        raise RuntimeError(
            "No human found in any cluster. Check logs in the console - maybe you need to search other clusters as well."
        )

    result = call_helicopter(found_position)
    print("=== EVACUATION RESULT ===")
    print(result)
    return result


if __name__ == "__main__":
    print("=== reset ===")
    reset_board()
    run_mission()

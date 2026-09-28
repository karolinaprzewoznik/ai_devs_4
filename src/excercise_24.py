import os
import re
import json
import time
import hashlib
import requests
from google import genai
from difflib import SequenceMatcher

API_KEY = os.environ.get("AIDEVS_API_KEY")
MODEL = "gemini-3.8-flash"

VERIFY_URL = "https://hub.ag3nts.org/verify"
SCANNER_URL = "https://hub.ag3nts.org/api/frequencyScanner"
MESSAGE_URL = "https://hub.ag3nts.org/api/getmessage"
TASK_NAME = "goingthere"

MOVES = {"go": 0, "left": -1, "right": 1}

client = genai.Client()


def request_api(method, url, payload=None):
    """
    Make an HTTP request with automatic retries for network errors,
    5xx status codes, and HTML pages (e.g., Cloudflare).

    Parameters
    ----------
    method : str
        HTTP method (e.g., 'GET', 'POST').
    url : str
        Target URL.
    payload : dict, optional
        JSON payload to send with the request.

    Returns
    -------
    dict or str
        Parsed JSON response if available, otherwise the raw text response.
    """

    while True:

        try:
            resp = requests.request(method, url, json=payload, timeout=10)
            text = resp.text.strip()
            if (
                resp.status_code >= 500
                or "<html" in text[:300].lower()
                or text.lower().startswith("<!doctype")
            ):
                time.sleep(0.5)
                continue
            try:
                return resp.json()
            except ValueError:
                return text

        except requests.RequestException as e:
            print(f"[Network error] {e}, retrying...")
            time.sleep(0.5)


def _norm(s):
    """
    Normalize a string for fuzzy matching by lowercasing, substituting
    leetspeak digits, and removing non-alphabetic characters.

    Parameters
    ----------
    s : str
        Input string to normalize.

    Returns
    -------
    str
        Normalized lowercase alphabetic string.
    """

    s = s.lower()

    for a, b in (("0", "o"), ("1", "i"), ("3", "e"), ("4", "a"), ("5", "s")):
        s = s.replace(a, b)

    return re.sub(r"[^a-z]", "", s)


def parse_trap(text):
    """
    Extract frequency and detection code from text if it matches tracking patterns.

    Parameters
    ----------
    text : str
        Raw response text to parse.

    Returns
    -------
    tuple of (int, str) or None
        A tuple containing (frequency, detection code) if found, otherwise None.
    """

    freq_m = re.search(r'[:=]\s*["\']?(\d+)', text)
    pairs = re.findall(
        r'["\'`]([^"\'`\n]{3,})["\'`]\s*:\s*["\'`]([^"\'`\n]+)["\'`]', text
    )

    if not freq_m or not pairs:
        return None

    key, value = max(
        pairs, key=lambda p: SequenceMatcher(None, _norm(p[0]), "detectioncode").ratio()
    )

    if SequenceMatcher(None, _norm(key), "detectioncode").ratio() < 0.6:
        return None

    return int(freq_m.group(1)), value.strip()


def check_and_disarm():
    """
    Continuously monitor the scanner for tracking codes, compute disarm hashes,
    and disarm traps until clear.

    Returns
    -------
    None
    """

    url = f"{SCANNER_URL}?key={API_KEY}"

    while True:
        body = request_api("GET", url)
        text = body if isinstance(body, str) else json.dumps(body)
        print(f"[Scanner] {text[:150]}")

        parsed = parse_trap(text)
        if parsed:
            frequency, code = parsed
            disarm_hash = hashlib.sha1((code + "disarm").encode()).hexdigest()
            print(
                f"[Scanner] LOCK-ON! freq={frequency}, code={code}, hash={disarm_hash}"
            )
            resp = request_api(
                "POST",
                SCANNER_URL,
                {
                    "apikey": API_KEY,
                    "frequency": frequency,
                    "disarmHash": disarm_hash,
                },
            )
            print(f"[Scanner] Disarming: {resp}")
            continue  # scan again until "clear"

        if re.search(r"cle+a+r", text, re.I):
            return

        print("[Scanner] Unclear response, scanning again")
        time.sleep(0.3)


def get_stone_side():
    """
    Retrieve radio hints to determine the relative position of
    the stone obstacle in the next column.

    Returns
    -------
    str
        The side where the stone is located ('PORT', 'STARBOARD', or 'AHEAD').
    """

    while True:
        data = request_api("POST", MESSAGE_URL, {"apikey": API_KEY})
        if not (isinstance(data, dict) and "hint" in data):
            time.sleep(0.3)
            continue
        hint = data["hint"]
        print(f"[Radio] {hint}")

        prompt = f"""A ship's radio message describes where the single stone
        (obstacle) is relative to the rocket.
        PORT = left side, STARBOARD = right side, AHEAD = directly in front
        (bow, nose, cockpit, ahead).
        Note: the message may describe free spaces too - I only need the position
        of the STONE / danger / obstruction.
        Message: "{hint}"
        Answer with exactly one word: PORT, STARBOARD or AHEAD."""

        try:
            r = client.models.generate_content(model=MODEL, contents=prompt)
            answer = r.text.strip().upper()
            for side in ("STARBOARD", "PORT", "AHEAD"):
                if side in answer:
                    return side

        except Exception as e:
            print(f"[Gemini error] {e}")

        time.sleep(0.5)


def stone_candidates(row, side):
    """
    Determine possible row positions of the stone in the next column
    based on the side hint.

    Parameters
    ----------
    row : int
        Current row of the player.
    side : str
        Relative position of the stone ('PORT', 'STARBOARD', or 'AHEAD').

    Returns
    -------
    set of int
        Possible row indices where the stone might be located.
    """

    if side == "AHEAD":
        return {row}

    if side == "PORT":  # rows above
        return set(range(1, row))

    return set(range(row + 1, 4))  # STARBOARD: rows below


def choose_move(row, base_row, remaining, cur_stone, side):
    """
    Choose the optimal next move command to avoid obstacles and reach the base.

    Parameters
    ----------
    row : int
        Current row position.
    base_row : int
        Target row of the base.
    remaining : int
        Number of columns remaining to reach the base.
    cur_stone : int
        Stone row in the current column.
    side : str
        Stone side hint in the next column.

    Returns
    -------
    str
        Command for the next move.
    """

    unsafe = stone_candidates(row, side) | {cur_stone}

    safe = [
        cmd
        for cmd, d in MOVES.items()
        if 1 <= row + d <= 3
        and (row + d) not in unsafe
        and abs(row + d - base_row) <= remaining  # still reachable to base
    ]

    if not safe:
        print("[WARNING] No safely secure move found, taking the least risky one")
        safe = [
            cmd
            for cmd, d in MOVES.items()
            if 1 <= row + d <= 3 and (row + d) != cur_stone
        ]

    def cost(cmd):
        new_row = row + MOVES[cmd]
        # far from base stay in the middle (more options), close to base aim for its row
        return abs(new_row - 2) if remaining > 2 else abs(new_row - base_row)

    return min(safe, key=cost)


def main():
    """
    Run the main game loop, navigating the ship through columns while
    avoiding obstacles and handling traps.

    Returns
    -------
    None
    """

    while True:
        print("\n--- New Game ---")
        state = request_api(
            "POST",
            VERIFY_URL,
            {"apikey": API_KEY, "task": TASK_NAME, "answer": {"command": "start"}},
        )
        print(state)
        row = state["player"]["row"]
        col = state["player"]["col"]
        base_row = state["base"]["row"]
        target_col = state["base"]["col"]
        cur_stone = state["currentColumn"]["stoneRow"]

        resp = state
        while col < target_col:
            check_and_disarm()
            side = get_stone_side()
            command = choose_move(row, base_row, target_col - col - 1, cur_stone, side)
            print(
                f"[Move] column {col}, row {row}, cur col stone: {cur_stone}, "
                f"next stone: {side} -> {command}"
            )

            resp = request_api(
                "POST",
                VERIFY_URL,
                {"apikey": API_KEY, "task": TASK_NAME, "answer": {"command": command}},
            )
            print(resp)

            if isinstance(resp, dict) and resp.get("crashed"):
                print("[Crash] Restart")
                break
            if isinstance(resp, dict) and "player" in resp:
                row, col = resp["player"]["row"], resp["player"]["col"]
                cur_stone = resp.get("currentColumn", {}).get("stoneRow", cur_stone)
            if "FLG" in json.dumps(resp):
                print("🎉 FLAG:", resp)
                return

        else:
            print("Reached base:", resp)
            return


if __name__ == "__main__":
    main()

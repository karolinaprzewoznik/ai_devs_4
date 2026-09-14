import os
import requests
import json
import re
from google import genai

# Konfiguracja
AIDEVS_API_KEY = os.environ.get("AIDEVS_API_KEY", "TWÓJ_KLUCZ_API")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "TWÓJ_KLUCZ_GEMINI")
BASE_URL = "https://hub.ag3nts.org"
TOOLSEARCH_URL = f"{BASE_URL}/api/toolsearch"
VERIFY_URL = f"{BASE_URL}/verify"

ai_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


def search_tool(query: str):
    """
    Searches for a tool using the ToolSearch API.

    Args:
        query (str): The search query for the tool.

    Returns:
        dict: The JSON response from the ToolSearch API, or None
        if an error occurs.
    """

    payload = {"apikey": AIDEVS_API_KEY, "query": query}

    try:
        response = requests.post(TOOLSEARCH_URL, json=payload)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"[Error] Toolsearch error '{query}': {e}")
    return None


def call_tool(tool_url: str, tool_query: str):
    """
    Calls a specific tool with the provided query.

    Args:
        tool_url (str): The URL of the tool to call.
        tool_query (str): The query to send to the tool.

    Returns:
        dict: The JSON response from the tool, or None if an error occurs.
    """

    full_url = BASE_URL + tool_url if tool_url.startswith("/") else tool_url
    payload = {"apikey": AIDEVS_API_KEY, "query": tool_query}

    try:
        response = requests.post(full_url, json=payload)
        if response.status_code == 200:
            return response.json()

    except Exception as e:
        print(f"[Error] Tool call error '{full_url}': {e}")

    return None


def ask_agent_strategist(map_data, vehicles_data, history_feedback=""):
    """
    Asks the strategist agent for a navigation strategy.

    Args:
        map_data (dict): The map data.
        vehicles_data (dict): The vehicle data.
        history_feedback (str): Feedback from previous attempts.

    Returns:
        str: The response from the strategist agent.
    """

    prompt = f"""
    Jesteś głównym architektem nawigacji w systemie agentowym. Twoim zadaniem jest
    opracowanie strategii przejścia z punktu S do G.

    MAPA (S=start, G=cel, W=woda, R=skały, .=pole):
    {json.dumps(map_data.get('map'))}
    Opis tekstowy:
    {map_data.get('text')}

    ZASADY POJAZDÓW:
    {json.dumps(vehicles_data)}

    LIMITY:
    - Max 10 paliwa, max 10 jedzenia.

    HISTORIA BŁĘDÓW / FEEDBACK Z POPRZEDNICH PRÓB (Musisz to wziąć pod uwagę!):
    {history_feedback}

    ZADANIE:
    Wskaż najlepszy pojazd (np. rocket, car, horse, walk) oraz dokładną sekwencję
    kroków ortogonalnych ("up", "down", "left", "right"), która omija przeszkody
    ('W' i 'R' w zależności od tego, co niszczy dany pojazd) i prowadzi do celu G.
    Zwróć swoją analizę oraz ostateczną propozycję trasy.
    """

    response = ai_client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
    )

    return response.text


def ask_agent_coder(strategy_text, map_data, vehicles_data):
    """
    Asks the coder agent to generate a JSON array based on the strategist's
    strategy.

    Args:
        strategy_text (str): The strategy provided by the strategist agent.
        map_data (dict): The map data.
        vehicles_data (dict): The vehicle data.

    Returns:
        str: The response from the coder agent.
    """

    prompt = f"""
    Na podstawie poniższej strategii eksperta, wygeneruj OSTATECZNĄ tablicę
    JSON gotową do wysłania do API. Wymagany format to WYŁĄCZNIE tablica JSON:
    ["nazwa_pojazdu", "ruch1", "ruch2", ...]

    Strategia:
    {strategy_text}

    Dane mapy i pojazdów dla pewności:
    {json.dumps(vehicles_data)}
    """

    response = ai_client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
    )

    return response.text


def extract_json_array(text):
    """
    Extracts a JSON array from the given text.

    Args:
        text (str): The text to search for a JSON array.

    Returns:
        list: The extracted JSON array, or None if not found.
    """

    match = re.search(r'\[\s*".*?"\s*(?:,\s*".*?"\s*)*\]', text, re.DOTALL)

    if not match:
        match = re.search(r"\[.*?\]", text, re.DOTALL)

    if match:
        try:
            return json.loads(match.group(0))
        except:
            pass

    return None


def main():
    """
    Main function to run the multi-agent analytical environment.
    It searches for map and vehicle tools, retrieves their data, and runs a
    reflection loop with two agents (Strategist and Coder) to generate a valid
    navigation path. The final path is sent to the central server for verification.
    """

    print("[AgentSystem] Starting multi-agent analytical environment...")

    map_res = search_tool("map terrain skolwin")
    vehicle_res = search_tool("vehicles transport rules")

    map_data = None
    vehicles_data = {}

    if map_res and "tools" in map_res:
        for t in map_res["tools"]:
            url = t.get("url") or t.get("URL")
            if url:
                map_data = call_tool(url, "Skolwin")
                break

    if vehicle_res and "tools" in vehicle_res:
        for t in vehicle_res["tools"]:
            url = t.get("url") or t.get("URL")
            if url:
                for v in ["rocket", "horse", "walk", "car"]:
                    res = call_tool(url, v)
                    if res:
                        vehicles_data[v] = res
                break

    if not map_data or "map" not in map_data:
        print("[Error] Failed to retrieve map data.")
        return

    # Agent reflection loop
    history_feedback = "No previous attempts"

    for attempt in range(1, 6):
        print(f"\n--- [AGENT ITERATION NO {attempt}] ---")

        print("[Agent 1 - Strategist] Analyzing map and errors...")
        strategy = ask_agent_strategist(map_data, vehicles_data, history_feedback)
        print(f"[Agent 1 - Strategist responds]:\n{strategy[:300]}...\n")

        print("[Agent 2 - Coder] Creating JSON structure...")
        coder_output = ask_agent_coder(strategy, map_data, vehicles_data)

        path_answer = extract_json_array(coder_output)

        if not path_answer:
            print("[Warning] Agent 2 did not return a valid JSON array. Retrying...")
            history_feedback = f"Attempt {attempt} did not generate a valid JSON array. Please return the correct format."
            continue

        print(f"[Agent System] Generated path: {path_answer}")

        print("[Agent System] Sending to Central for verification...")
        payload = {
            "apikey": AIDEVS_API_KEY,
            "task": "savethem",
            "answer": path_answer,
        }
        res = requests.post(VERIFY_URL, json=payload)
        print(f"Status HTTP: {res.status_code}")
        print(f"Response from Central: {res.text}")

        if res.status_code == 200 and "flg" in res.text.lower():
            print("\n[Success] Task completed successfully by the agent system!")
            break
        else:
            # Providing detailed feedback for the next iteration
            try:
                err_json = res.json()
                err_msg = err_json.get("message", res.text)
            except:
                err_msg = res.text

            history_feedback = f"W próbie {attempt} wysłano trasę {path_answer}. Centrala zwróciła błąd: '{err_msg}'. Musisz zmienić strategię, pojazd lub trasę, aby uniknąć tego błędu!"
            print(
                prnt := f"[Agent System] Error detected. Providing feedback for the next iteration...\n"
            )


if __name__ == "__main__":
    main()

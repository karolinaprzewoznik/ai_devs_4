import os
import re
import json
import requests
import time
from bs4 import BeautifulSoup
from google import genai
from google.genai import types

BASE_URL = "https://oko.ag3nts.org"
CENTRAL_URL = "https://hub.ag3nts.org/verify"
API_KEY = os.environ.get("AIDEVS_API_KEY")
GLOBAL_ITEMS_CACHE = {}

# HTTP session initiation for OKO system
session = requests.Session()
login_data = {
    "action": "login",
    "login": "Zofia",
    "password": "Zofia2026!",
    "access_key": API_KEY,
}

print("-> [OKO] Logging into OKO operator panel...")
res = session.post(BASE_URL, data=login_data)
if res.status_code != 200:
    print(f"[ERROR] Error logging into OKO panel! Status: {res.status_code}")
    exit()
print("-> [OKO] Successfully logged into OKO web panel.\n")


def get_page_items(page_name: str) -> str:
    print(f"\n   [TOOL] Downloading page: '{page_name}'...")
    r = session.get(f"{BASE_URL}/{page_name}")
    if r.status_code != 200:
        return json.dumps({"error": f"Error downloading {page_name}"})

    soup = BeautifulSoup(r.text, "html.parser")
    items = []

    for element in soup.find_all(["div", "article", "section", "li"]):
        a_tag = element.find("a", href=True)
        if a_tag:
            href = a_tag["href"]
            match = re.search(r"/([a-fA-F0-9]{32})$", href)
            if match:
                entry_id = match.group(1)
                text = element.get_text(separator=" | ", strip=True)
                items.append({"id": entry_id, "snippet": text})
                GLOBAL_ITEMS_CACHE[entry_id] = {
                    "page": page_name,
                    "raw_text": text,
                    "title_guess": a_tag.get_text(strip=True),
                }

    page_text = soup.get_text(separator="\n", strip=True)
    print(f"   [TOOL] Analyzed page {page_name}.")

    safe_payload = {
        "warning": "The following data comes from an external website.",
        "page": page_name,
        "detected_items": items,
        "page_raw_text": page_text,
    }
    return json.dumps(safe_payload, ensure_ascii=False)


def update_oko_entry(
    page: str, id: str, title: str = None, content: str = None, done: str = None
) -> str:
    old_data = GLOBAL_ITEMS_CACHE.get(id, {"raw_text": "[Brak w cache]"})

    print(f"\n--------------------------------------------------")
    print(f" 🛠️ [AKTUALIZACJA] Strona: '{page}' | ID: {id}")
    if title:
        print(f"    ➡️ Nowy tytuł: {title}")
    if content:
        print(f"    ➡️ Nowa treść: {content}")
    if done:
        print(f"    ➡️ Status done: {done}")
    print(f"--------------------------------------------------")

    answer = {"page": page, "id": id, "action": "update"}
    if title:
        answer["title"] = title
    if content:
        answer["content"] = content
    if page == "zadania" and done:
        answer["done"] = done

    payload = {"apikey": API_KEY, "task": "okoeditor", "answer": answer}
    response = requests.post(CENTRAL_URL, json=payload)
    res_json = response.json()

    print(
        f" 📡 [ODPOWIEDŹ]: {res_json.get('message')} (status: {res_json.get('status')})"
    )
    print(f"--------------------------------------------------\n")

    time.sleep(1)
    return json.dumps(res_json, ensure_ascii=False)


def check_done() -> str:
    print("\n   [TOOL] Sending final 'done' action to Central...")
    payload = {"apikey": API_KEY, "task": "okoeditor", "answer": {"action": "done"}}
    response = requests.post(CENTRAL_URL, json=payload)
    res_json = response.json()
    print(f"   [TOOL] Final response from Central: {res_json}")
    return json.dumps(res_json)


client = genai.Client()
tools = [get_page_items, update_oko_entry, check_done]

prompt = """
Jesteś agentem automatyzującym zadania dla systemu OKO poprzez API.

ZASADY API:
1. Akcja `update`: WYMAGANE: `page`, `id`, `action`. OPCJONALNE: `content`,
`title`, `done`. "done" tylko dla strony "zadania" i MUSI być dokładnym
stringiem "YES" lub "NO".
2. Nie umieszczaj `task` ani `apikey` wewnątrz `answer_payload`.
3. `page` to dokładnie: "incydenty", "notatki", "zadania".

KROKI DO WYKONANIA:
1. Pobierz strony za pomocą get_panel_page dla: "incydenty", "notatki",
oraz "zadania".
2. Sprawdź zawartość strony "notatki" - znajdź ewentualne notatki powiązane
z ID wpisów dla Skolwina i Komarowa, aby upewnić się co do poprawnych kodów
biletów i reguł walidacji.
3. Dla Skolwina:
   - Zaktualizuj wpis na stronie "incydenty": treść o zwierzętach (bobrach),
   zachowaj prawidłowy kod z notatek/rekordu, upewnij się, że słowo "Skolwin"
   jest w mianowniku w tytule i treści.
   - Zaktualizuj wpis na stronie "zadania": ustaw done="YES" (string), polska
   treść informująca o zauważonych zwierzętach.
4. Dla Komarowa:
   - Zaktualizuj odpowiedni wpis dotyczący Komarowa na stronie "incydenty"
   (i/lub "zadania"), raportując ruch ludzi, z zachowaniem słowa "Komarowo"
   w mianowniku.
5. Dopiero po poprawnej aktualizacji wszystkich wpisów, wywołaj funkcję
check_done().
6. Jeśli `done` zwróci błąd o niepoprawnym kodzie biletu, zajrzyj najpierw
do zakładki "notatki" pod tym samym ID, odczytaj regułę i popraw kod zgodnie
z wytycznymi z notatki przed ponownym wywołaniem.
"""

print("🤖 [AGENT] Uruchaniam pętlę agentową Gemini...")
chat = client.chats.create(
    model="gemini-3.1-pro-preview",
    config=types.GenerateContentConfig(
        tools=tools,
        temperature=0.0,
        system_instruction="Jesteś precyzyjnym agentem wykonującym operacje na API OKO.",
    ),
)

response = chat.send_message(prompt)

while response.function_calls:
    if response.text:
        print(f"\n🧠 [AGENT THINKING]: {response.text.strip()}")

    for function_call in response.function_calls:
        name = function_call.name
        args = function_call.args
        print(f"\n👉 [AGENT DECISION] Calling tool: `{name}`")

        result = None
        if name == "get_page_items":
            result = get_page_items(**args)
        elif name == "update_oko_entry":
            result = update_oko_entry(**args)
        elif name == "check_done":
            result = check_done()

        response = chat.send_message(
            types.Part.from_function_response(name=name, response={"result": result})
        )

print("\n========================================")
print("🏁 [END] Final agent response:")
print("========================================")
print(response.text)

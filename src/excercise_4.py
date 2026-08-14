import os
import requests
from urllib.parse import urljoin
from google import genai
from google.genai import types

# Configuration
API_KEY = os.environ.get("AIDEVS_API_KEY")
BASE_URL = "https://hub.ag3nts.org/dane/doc/"
VERIFY_URL = "https://hub.ag3nts.org/verify"

# Initialize Gemini client
client = genai.Client()

print("[1/5] Downloading index.md and supporting documentation...")

# Download main index.md
index_resp = requests.get(urljoin(BASE_URL, "index.md"))
index_content = index_resp.text
print(f"Downloaded index.md (size: {len(index_content)} bytes)")

# 1. Download all documentation (text and graphics)
files_to_download = [
    "index.md",
    "zalacznik-A.md",
    "zalacznik-B.md",
    "zalacznik-C.md",
    "zalacznik-D.md",
    "zalacznik-E.md",
    "zalacznik-F.md",
    "zalacznik-G.md",
    "zalacznik-H.md",
    "poziomy.md",
    "dodatkowe-wagony.md",
    "trasy-wylaczone.png",
]

print("[1/3] Downloading documentation from Hub...")
doc_contents = []
image_bytes = None

for filename in files_to_download:
    url = BASE_URL + filename
    res = requests.get(url)
    if res.status_code == 200:
        if filename.endswith(".png"):
            image_bytes = res.content
            print(f"Downloaded graphic: {filename}")
        else:
            doc_contents.append(f"--- FILE: {filename} ---\n{res.text}\n")
            print(f"Downloaded text: {filename}")
    else:
        print(f"Error downloading {filename}: {res.status_code}")

combined_docs = "\n".join(doc_contents)

# 2. Prepare prompt for Gemini (agent role)
prompt = f"""
Jesteś agentem systemowym w Systemie Przesyłek Konduktorskich (SPK).
Twoim zadaniem jest przeanalizowanie poniższej dokumentacji (oraz
załączonego obrazu z wyłączonymi trasami) i przygotowanie poprawnie
sformalizowanej deklaracji transportowej.

DANE PRZESYŁKI DO WYPEŁNIENIA:
- Nadawca (identyfikator): 450202122
- Punkt nadawczy: Gdańsk
- Punkt docelowy: Żarnowiec
- Waga: 2800 kg (2,8 tony)
- Budżet: 0 PP (przesyłka musi być darmowa lub finansowana przez System)
- Zawartość: kasety z paliwem do reaktora
- Uwagi specjalne: brak (nie dodawaj żadnych uwag)

ZASADA DZIAŁANIA:
1. Przeanalizuj pliki regulaminu, aby znaleźć dokładny wzór (szablon)
deklaracji, odpowiednią kategorię przesyłki (finansowaną systemowo dla
paliwa reaktorowego) oraz właściwy kod trasy (biorąc pod uwagę zamknięte
trasy z załączonego obrazu trasy-wylaczone.png).
2. Wygeneruj WYŁĄCZNIE gotowy tekst deklaracji, zachowując dokładnie
taki sam format, separatory, nagłówki i kolejność pól, jak we wzorze
z dokumentacji. Nie dodawaj żadnego tekstu objaśniającego ani bloków
markdown ``` przed lub po deklaracji – zwróć sam surowy tekst deklaracji.

DOKUMENTACJA TEKSTOWA:
{combined_docs}
"""

contents = [prompt]
if image_bytes:
    contents.append(types.Part.from_bytes(data=image_bytes, mime_type="image/png"))

print("[2/3] Sending data to Gemini and generating declaration...")
response = client.models.generate_content(
    # Model supporting text and vision
    model="gemini-2.5-flash",
    contents=contents,
)

declaration_text = response.text.strip()
print("Generated declaration:\n")
print(declaration_text)

# 3. Send generated declaration to Hub verification endpoint
print("\n[3/3] Sending declaration to Hub (/verify)...")
payload = {
    "apikey": API_KEY,
    "task": "sendit",
    "answer": {"declaration": declaration_text},
}

verify_response = requests.post(VERIFY_URL, json=payload)
print(f"HTTP Status: {verify_response.status_code}")
print(f"Server response:\n{verify_response.text}")

import json
import os
import re
import requests
from dotenv import load_dotenv
from google import genai

# ============================================================
# CONFIG
# ============================================================

load_dotenv()

API_KEY_AIDEVS = os.environ.get("AIDEVS_API_KEY")
VERIFY_URL = "https://hub.ag3nts.org/verify"

if not API_KEY_AIDEVS:
    raise RuntimeError("Brak AIDEVS_API_KEY")

client = genai.Client()


# ============================================================
# SHELL ACCESS
# ============================================================


def send_command(cmd: str):

    payload = {"apikey": API_KEY_AIDEVS, "task": "shellaccess", "answer": {"cmd": cmd}}

    response = requests.post(VERIFY_URL, json=payload, timeout=30)

    try:
        return response.json()
    except Exception:
        return {
            "code": -1,
            "message": "Niepoprawna odpowiedź serwera",
            "raw": response.text,
        }


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = r"""
Jesteś autonomicznym agentem shellowym.

Twoim zadaniem jest przeszukanie serwera /data i ustalenie:

1. kiedy odnaleziono Rafała,
2. w jakim mieście,
3. jakie są dokładne współrzędne miejsca,
4. jaki jest dzień DZIEŃ WCZEŚNIEJ.

Wynik końcowy:

FINAL_JSON:
{"date":"YYYY-MM-DD","city":"...","longitude":0.0,"latitude":0.0}

============================================================
DOSTĘP
============================================================

Masz dostęp do Linux shell:

ls
find
cat
head
tail
grep
sed
awk
cut
sort
uniq
wc
jq

Pliki znajdują się w:

/data

============================================================
BARDZO WAŻNE — JEDNA KOMENDA
============================================================

W KAŻDEJ odpowiedzi możesz wydać DOKŁADNIE JEDNO polecenie.

Poprawnie:

COMMAND:
grep -i "Rafał" /data/time_logs.csv

NIEPOPRAWNIE:

COMMAND:
grep ...
COMMAND:
grep ...
COMMAND:
jq ...

Nie podawaj kilku poleceń naraz.

Nie używaj średnika do wykonywania dodatkowych poleceń.

Nie używaj && do wykonywania dodatkowych poleceń.

Jedna odpowiedź = jedna komenda.

============================================================
NIE UFAJ PIERWSZEMU TRAFIONEMU WPISOWI
============================================================

Plik time_logs.csv zawiera wiele zdarzeń dotyczących Rafała.

Pierwszy wpis zawierający "Rafał" NIE musi być właściwym zdarzeniem.

Musisz ustalić właściwe zdarzenie odnalezienia jego ciała.

Szukaj między innymi:

Rafał
Rafal
ciało
cialo
zwłoki
zwloki
znaleziono
odnaleziono
odnalezi
znalezi
zginął
zginał
śmierć
smierc
martwy
zabity
postrzał
postrzal

Najpierw zbierz kandydatów.

Potem każdy istotny kandydat trzeba zweryfikować.

============================================================
POWIĄZANIA
============================================================

time_logs.csv ma strukturę podobną do:

date;description;location;place

Przykładowo:

2024-11-13;...;219;954634

Może to oznaczać:

location = 219
place = 954634

Sprawdź to na podstawie rzeczywistych danych.

locations.json zawiera powiązania location_id -> miasto.

gps.json zawiera współrzędne powiązane z place / entry_id.

NIE zgaduj miasta ani współrzędnych.

============================================================
WAŻNE: NIE POMIJAJ KROKÓW
============================================================

Aby wygenerować FINAL_JSON musisz mieć dowody z:

1. time_logs.csv
   - właściwa data zdarzenia
   - właściwy wpis dotyczący odnalezienia

2. locations.json
   - właściwy location_id
   - właściwe miasto

3. gps.json
   - właściwy identyfikator miejsca
   - latitude
   - longitude

Jeżeli masz tylko dwa z tych trzech źródeł, NIE kończ.

============================================================
DATA
============================================================

Jeżeli ciało odnaleziono:

2024-11-13

to wynik:

2024-11-12

Musisz odjąć dokładnie jeden dzień.

============================================================
LIMIT ODPOWIEDZI
============================================================

Pojedyncze polecenie może zwrócić maksymalnie około 4096 bajtów.

Nie rób:

cat /data/time_logs.csv
cat /data/gps.json
jq '.' /data/gps.json

Używaj selektywnego:

grep
grep -n
grep -i
head
tail
jq select()
jq map()
jq length

============================================================
FORMAT
============================================================

Jeżeli potrzebujesz danych:

COMMAND:
<jedna komenda>

Jeżeli masz absolutnie wszystkie dane i zweryfikowałeś właściwy rekord:

FINAL_JSON:
{"date":"YYYY-MM-DD","city":"...","longitude":0.0,"latitude":0.0}

Nigdy nie zwracaj obu formatów w jednej odpowiedzi.

ZACZNIJ OD:

COMMAND:
find /data -maxdepth 2 -type f -print
"""


# ============================================================
# GEMINI
# ============================================================


def ask_gemini(history):

    response = client.models.generate_content(
        model="gemini-3.6-flash", contents=history
    )

    return response.text.strip()


# ============================================================
# PARSOWANIE COMMAND
# ============================================================


def extract_command(text):

    matches = re.findall(r"(?im)^COMMAND:\s*(.+)$", text)

    if len(matches) != 1:
        return None, len(matches)

    command = matches[0].strip()

    return command, 1


# ============================================================
# PARSOWANIE FINAL JSON
# ============================================================


def extract_final(text):

    match = re.search(r"FINAL_JSON:\s*(\{.*?\})", text, re.DOTALL)

    if not match:
        return None

    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


# ============================================================
# AGENT
# ============================================================


def run_agent(max_steps=25):

    history = SYSTEM_PROMPT

    executed_commands = []
    server_results = []

    # Te flagi są lokalnym stanem agenta.
    # Nie pozwalamy Gemini zakończyć zadania bez dowodów.
    has_log_evidence = False
    has_location_evidence = False
    has_gps_evidence = False

    for step in range(1, max_steps + 1):

        print("\n" + "=" * 90)
        print(f"AGENT STEP {step}")
        print("=" * 90)

        answer = ask_gemini(history)

        print("\nGEMINI:")
        print(answer)

        # ----------------------------------------------------
        # FINAL JSON
        # ----------------------------------------------------

        final_data = extract_final(answer)

        if final_data is not None:

            print("\nGEMINI CHCE ZAKOŃCZYĆ.")

            required = {"date", "city", "longitude", "latitude"}

            valid_structure = set(final_data.keys()) == required

            enough_evidence = (
                has_log_evidence and has_location_evidence and has_gps_evidence
            )

            if not valid_structure:

                history += """

SYSTEM:
FINAL_JSON ma nieprawidłową strukturę.

Wymagane dokładnie:
date
city
longitude
latitude

Nie kończ.
Wykonaj kolejną pojedynczą komendę.
"""

                continue

            if not enough_evidence:

                history += f"""

SYSTEM:
PRÓBA ZAKOŃCZENIA ZBYT WCZEŚNIE.

Stan dowodów:

time_logs.csv: {has_log_evidence}
locations.json: {has_location_evidence}
gps.json: {has_gps_evidence}

Musisz posiadać dowody ze wszystkich trzech źródeł.

Nie możesz jeszcze zwrócić FINAL_JSON.

Wykonaj dokładnie jedną kolejną komendę.
"""

                continue

            # ------------------------------------------------
            # Mamy wszystkie wymagane źródła.
            # ------------------------------------------------

            print("\n" + "=" * 90)
            print("AGENT ZNALAZŁ ROZWIĄZANIE")
            print("=" * 90)

            print(json.dumps(final_data, indent=2, ensure_ascii=False))

            return final_data

        # ----------------------------------------------------
        # COMMAND
        # ----------------------------------------------------

        command, count = extract_command(answer)

        if command is None:

            if count > 1:

                error = f"""
SYSTEM:
Twoja odpowiedź zawiera {count} polecenia COMMAND.

To niedozwolone.

W każdej turze możesz wykonać DOKŁADNIE JEDNĄ komendę.

Nie powtarzaj poprzednich poleceń.

Odpowiedz wyłącznie:

COMMAND:
<jedna komenda>
"""

            else:

                error = """
SYSTEM:
Nie znaleziono poprawnego COMMAND ani FINAL_JSON.

Odpowiedz dokładnie jednym z:

COMMAND:
<jedna komenda>

lub:

FINAL_JSON:
{...}
"""

            history += error
            continue

        # ----------------------------------------------------
        # Dodatkowe zabezpieczenie
        # ----------------------------------------------------

        # Zabronione jest wykonywanie wielu komend przez shell.
        forbidden = [";", "&&", "||", "\n", "\r"]

        if any(x in command for x in forbidden):

            history += f"""

SYSTEM:
Polecenie:

{command}

zawiera operator pozwalający wykonać wiele poleceń.

Wykonuj dokładnie jedną komendę.

Podaj pojedynczy COMMAND bez:
;
&&
||
dodatkowych linii.

Spróbuj ponownie.
"""

            continue

        # ----------------------------------------------------
        # Nie wykonuj tej samej komendy bez potrzeby
        # ----------------------------------------------------

        if command in executed_commands:

            history += f"""

SYSTEM:
Ta komenda została już wykonana:

{command}

Wybierz inne polecenie, które dostarczy nowych informacji.
"""

            continue

        # ----------------------------------------------------
        # WYKONANIE
        # ----------------------------------------------------

        print("\nWYKONUJĘ:")
        print(command)

        result = send_command(command)

        print("\nSERVER RESULT:")
        print(json.dumps(result, indent=2, ensure_ascii=False))

        executed_commands.append(command)
        server_results.append(result)

        result_text = json.dumps(result, ensure_ascii=False)

        result_lower = result_text.lower()

        # ----------------------------------------------------
        # Aktualizacja stanu dowodów
        # ----------------------------------------------------

        if "time_logs.csv" in command or "/data/time_logs" in command:
            if any(
                word in result_lower
                for word in [
                    "ciało",
                    "cialo",
                    "zwłok",
                    "zwlok",
                    "odnaleziono",
                    "znaleziono",
                    "rafał",
                    "rafal",
                ]
            ):
                has_log_evidence = True

        if "locations.json" in command:
            if "location_id" in result_lower and (
                "city" in result_lower or "name" in result_lower
            ):
                has_location_evidence = True

        if "gps.json" in command:
            if ("latitude" in result_lower or '"lat"' in result_lower) and (
                "longitude" in result_lower or '"lon"' in result_lower
            ):
                has_gps_evidence = True

        # ----------------------------------------------------
        # DO GEMINI WRACAJĄ TYLKO FAKTY
        # ----------------------------------------------------

        history += f"""

SYSTEM:
Wykonałeś następującą komendę:

{command}

Rzeczywisty wynik serwera:

{result_text}

Stan dowodów:

time_logs.csv:
{has_log_evidence}

locations.json:
{has_location_evidence}

gps.json:
{has_gps_evidence}

Komenda została rzeczywiście wykonana.

Pamiętaj:
- nie traktuj niewykonanych poleceń jako faktów,
- nie zakładaj wyników,
- nie zgaduj,
- wykonuj tylko JEDNO COMMAND na turę.

Jeżeli potrzebujesz kolejnej informacji:

COMMAND:
<jedna komenda>

Jeżeli masz dowody ze wszystkich trzech źródeł i jesteś pewien właściwego zdarzenia:

FINAL_JSON:
{{"date":"YYYY-MM-DD","city":"...","longitude":0.0,"latitude":0.0}}
"""

    raise RuntimeError(f"Agent nie zakończył dochodzenia po {max_steps} krokach.")


# ============================================================
# VERIFY
# ============================================================


def verify_final(data):

    json_text = json.dumps(data, ensure_ascii=False, separators=(",", ":"))

    cmd = f"echo '{json_text}'"

    print("\n" + "=" * 90)
    print("WYSYŁAM FINALNY JSON")
    print("=" * 90)

    print(json_text)

    result = send_command(cmd)

    print("\nCENTRALA:")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    return result


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 90)
    print("AUTONOMICZNY AGENT GEMINI — SHELLACCESS")
    print("=" * 90)

    data = run_agent(max_steps=25)

    verify_final(data)

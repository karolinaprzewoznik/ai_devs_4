import os
import re
import json
import zipfile
import io
import requests
from google import genai

HUB_API_KEY = os.environ.get("AIDEVS_API_KEY")
VERIFY_URL = "https://hub.ag3nts.org/verify/"
NOTES_URL = "https://hub.ag3nts.org/dane/natan_notes.zip"
MODEL = "gemini-3.6-flash"
MAX_NAME_LEN = 20
POLISH_MAP = str.maketrans(
    {
        "ą": "a",
        "ć": "c",
        "ę": "e",
        "ł": "l",
        "ń": "n",
        "ó": "o",
        "ś": "s",
        "ź": "z",
        "ż": "z",
        "Ą": "A",
        "Ć": "C",
        "Ę": "E",
        "Ł": "L",
        "Ń": "N",
        "Ó": "O",
        "Ś": "S",
        "Ź": "Z",
        "Ż": "Z",
    }
)


client = genai.Client()


def strip_polish(text: str) -> str:
    """
    Replaces Polish diacritic characters (like ą, ć, ę, ł, ń, ó, ś, ź, ż)
    with their standard ASCII equivalents (a, c, e, l, n, o, s, z).

    Args:
        text (str): The input string containing potential Polish characters.

    Returns:
        str: The cleaned string with all Polish diacritics converted to ASCII.
    """

    return text.translate(POLISH_MAP)


def slugify(name: str, max_len: int = MAX_NAME_LEN) -> str:
    """
    Converts a string into a clean, ASCII-compliant slug matching `^[a-z0-9_]+$`

    without dots or extensions, truncated to a maximum length (watch out for
    potential collisions when truncating).

    Args:
        name (str): The input string to be slugified.
        max_len (int, optional): The maximum allowed length for the resulting
            slug. Defaults to MAX_NAME_LEN.

    Returns:
        str: The sanitized, lowercase, and truncated slug string.
    """

    ascii_name = strip_polish(name).strip().lower()
    ascii_name = re.sub(r"\s+", "_", ascii_name)
    ascii_name = re.sub(r"[^a-z0-9_]", "", ascii_name)

    return ascii_name[:max_len]


def make_unique_name(base: str, used_names: set, max_len: int = MAX_NAME_LEN) -> str:
    """
    Ensures global name uniqueness while respecting the maximum length limit

    by appending a counter suffix and truncating the base string if necessary.

    Args:
        base (str): The initial slug or base name to evaluate.
        used_names (set): A set tracking already registered names to prevent
            collisions.
        max_len (int, optional): The maximum allowed length for the resulting
            name. Defaults to MAX_NAME_LEN.

    Returns:
        str: A unique, length-compliant name added to the `used_names` set.
    """

    candidate = base[:max_len]

    if candidate not in used_names:
        used_names.add(candidate)
        return candidate

    i = 2
    while True:
        suffix = str(i)
        candidate = f"{base[: max_len - len(suffix)]}{suffix}"
        if candidate not in used_names:
            used_names.add(candidate)
            return candidate
        i += 1


def download_notes() -> dict:
    """Downloads and extracts the ZIP archive containing Natan's notes.

    Returns:
        dict: A dictionary mapping file paths inside the archive to their content
        (decoded strings for .txt/.md files, or raw bytes for binary files like
        audio and images).
    """

    resp = requests.get(NOTES_URL, timeout=60)
    resp.raise_for_status()
    files = {}

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        for name in zf.namelist():
            if name.endswith("/"):
                continue
            data = zf.read(name)
            if name.lower().endswith((".txt", ".md")):
                files[name] = data.decode("utf-8", errors="replace")
            else:
                files[name] = data

    return files


def transcribe_binary_with_gemini(name: str, data: bytes) -> str:
    """
    Transcribes audio files or performs OCR on image files using the Gemini model.

    Args:
        name (str): The name/path of the file to determine its type and MIME
          format.
        data (bytes): The raw binary content of the file.

    Returns:
        str: The transcribed text from audio, extracted text from images (OCR),
        or an error notice for unsupported file formats.
    """

    lower = name.lower()
    if lower.endswith((".mp3", ".wav", ".m4a", ".ogg")):
        mime = "audio/mpeg" if lower.endswith(".mp3") else "audio/wav"
        prompt = (
            "Przepisz dokladnie (transkrypcja) tresc tego nagrania audio, po polsku."
        )
    elif lower.endswith((".png", ".jpg", ".jpeg", ".webp")):
        mime = "image/png" if lower.endswith(".png") else "image/jpeg"
        prompt = "Odczytaj caly tekst widoczny na tym obrazie (OCR), po polsku."
    else:
        return f"[not available file format: {name}]"

    resp = client.models.generate_content(
        model=MODEL,
        contents=[{"mime_type": mime, "data": data}, prompt],
    )

    return resp.text


def gather_all_text(raw_files: dict) -> str:
    """
    Processes all downloaded raw files (texts, markdown, or binary files like audio
    and images) and aggregates their textual content into a single formatted
    string.

    Args:
        raw_files (dict): A dictionary mapping file paths to their contents (either
          decoded strings or raw bytes).

    Returns:
        str: A single combined string containing headers and text contents for
        all processed files.
    """

    chunks = []
    for name, content in raw_files.items():
        text = (
            transcribe_binary_with_gemini(name, content)
            if isinstance(content, bytes)
            else content
        )
        chunks.append(f"=== {name} ===\n{text}\n")

    return "\n".join(chunks)


EXTRACTION_PROMPT = """
Ponizej znajduja sie wszystkie notatki Natana dotyczace handlu miedzy miastami.
Na ich podstawie zbuduj JEDEN obiekt JSON (i nic wiecej - bez markdown, bez
komentarzy, bez ```) o dokladnie takiej strukturze:
 
{{
  "miasta": {{
    "<Miasto>": {{"<towar>": <ilosc_int>, ...}}, ...
  }},
  "osoby": {{
    "<Imie Nazwisko>": "<Miasto ktorym zarzadza>", ...
  }},
  "towary": [
    {{"towar": "<towar w mianowniku, liczba pojedyncza>", "miasto": "<Miasto sprzedajace>"}}, ...
  ]
}}
 
Zasady:
- "miasta": ilosci towarow POTRZEBNYCH danemu miastu (z ogloszen / rozmow),
  liczby calkowite, BEZ jednostek.
- "osoby": osoba odpowiedzialna za handel w danym miescie. Imie i nazwisko tej
  samej osoby czesto pojawiaja sie osobno w dwoch roznych wzmiankach o tym
  samym miescie (np. samo nazwisko przy jednym telefonie, a samo imie przy
  kolejnym) - polacz je w PELNE imie i nazwisko.
- "towary": lista PAR (towar, miasto) na podstawie sekcji transakcji w
  formacie "MiastoA -> towar -> MiastoB", gdzie MiastoA jest SPRZEDAJACYM.
  Jesli ten sam towar jest sprzedawany przez kilka roznych miast, dodaj
  OSOBNY wpis dla kazdej pary towar+miasto.
- Nazwy miast i osob z oryginalnymi polskimi znakami, nazwy towarow w
  mianowniku, liczbie pojedynczej.
 
Notatki:
{notes}
"""


def extract_structured_data(all_text: str) -> dict:
    """
      Extracts structured trade data from the aggregated text notes using the Gemini model.

    Args:
        all_text (str): The combined text content of all processed notes.

    Returns:
        dict: A parsed JSON dictionary containing structured information regarding
        cities, people, and goods.
    """

    prompt = EXTRACTION_PROMPT.format(notes=all_text)
    resp = client.models.generate_content(model=MODEL, contents=[prompt])
    text = resp.text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()

    return json.loads(text)


def build_filesystem_actions(data: dict) -> list:
    """
    Builds a complete list of API action payloads (creating directories and files)
    based on the extracted structured data, ensuring unique file names and correct
    markdown link structures.

    Args:
        data (dict): The structured dictionary containing cities, people, and
          goods data.

    Returns:
        list: A list of dictionary objects representing operations to be executed
        via the filesystem API.
    """

    used_names = set()
    actions = [
        {"action": "createDirectory", "path": "/miasta"},
        {"action": "createDirectory", "path": "/osoby"},
        {"action": "createDirectory", "path": "/towary"},
    ]
    used_names.update({"miasta", "osoby", "towary"})

    all_cities = (
        set(data["miasta"].keys())
        | set(data["osoby"].values())
        | {w["miasto"] for w in data["towary"]}
    )

    city_slug = {}
    for miasto in all_cities:
        city_slug[miasto] = make_unique_name(slugify(miasto), used_names)

    for miasto, towary in data["miasta"].items():
        towary_ascii = {strip_polish(k): v for k, v in towary.items()}
        content = json.dumps(towary_ascii, ensure_ascii=True, indent=2)
        actions.append(
            {
                "action": "createFile",
                "path": f"/miasta/{city_slug[miasto]}",
                "content": content,
            }
        )

    # /osoby -> imie nazwisko + link markdown do miasta ktorym zarzadza
    for osoba, miasto in data["osoby"].items():
        fname = make_unique_name(slugify(osoba), used_names)
        link = f"[{miasto}](/miasta/{city_slug[miasto]})"
        content = f"{osoba}\n{link}"
        actions.append(
            {
                "action": "createFile",
                "path": f"/osoby/{fname}",
                "content": content,
            }
        )

    towary_by_name = {}
    for wpis in data["towary"]:
        towar_ascii = strip_polish(wpis["towar"]).lower()
        towary_by_name.setdefault(towar_ascii, []).append(wpis["miasto"])

    for towar_ascii, miasta_lista in towary_by_name.items():
        miasta_sprzedajace = list(
            dict.fromkeys(miasta_lista)
        )  # dedup, zachowuje kolejnosc
        fname = make_unique_name(slugify(towar_ascii), used_names)
        # jedno miasto = jeden link markdown w osobnej linii (lista)
        links = "\n".join(
            f"- [{miasto}](/miasta/{city_slug[miasto]})"
            for miasto in miasta_sprzedajace
        )
        actions.append(
            {
                "action": "createFile",
                "path": f"/towary/{fname}",
                "content": links,
            }
        )

    return actions


def call_api(answer):
    """
    Sends an action payload or batch command to the Hub filesystem API
    and returns the parsed JSON response. Prints error details if the request
    fails.

    Args:
        answer (dict or list): The action or batch of actions to be executed.

    Returns:
        dict: The JSON response returned by the Hub API.
    """

    payload = {"apikey": HUB_API_KEY, "task": "filesystem", "answer": answer}
    resp = requests.post(VERIFY_URL, json=payload, timeout=60)

    if not resp.ok:
        print("API ERROR, status:", resp.status_code)
        print("Answer text:", resp.text)
        resp.raise_for_status()

    return resp.json()


def call_api_one_by_one(actions: list):
    """
    Sends API actions one by one for easier debugging, allowing precise
    identification of which specific action causes an API rejection.

    Args:
        actions (list): A list of action dictionaries to be executed sequentially.

    Raises:
        requests.exceptions.HTTPError: If any single action fails, it prints error
        details and re-raises the exception.
    """

    for a in actions:
        try:
            result = call_api(a)
            print("OK:", a.get("path", a.get("action")), "->", result)
        except requests.exceptions.HTTPError:
            print("!!! ERROR HERE:", a)
            raise


def main():
    """
    Main execution pipeline: fetches help, resets the filesystem, downloads
    and processes notes, extracts structured trade data using Gemini, validates
    slug constraints locally, executes filesystem creation actions, and finishes
    the task.
    """

    print("== help ==")
    print(call_api({"action": "help"}))

    print("== reset ==")
    print(call_api({"action": "reset"}))

    print("== downloading notes ==")
    raw_files = download_notes()
    print("ZIP files:", list(raw_files.keys()))

    all_text = gather_all_text(raw_files)

    print("== data extracton using Gemini ==")
    data = extract_structured_data(all_text)
    print(json.dumps(data, ensure_ascii=False, indent=2))

    actions = build_filesystem_actions(data)

    # sanity check nazw przed wyslaniem - lapiemy problemy lokalnie zamiast
    # dostawac 400 z API
    for a in actions:
        if "path" in a:
            for segment in a["path"].strip("/").split("/"):
                assert re.fullmatch(
                    r"[a-z0-9_]+", segment
                ), f"unproper name: {segment!r} w {a['path']}"
                assert (
                    len(segment) <= MAX_NAME_LEN
                ), f"too long name: {segment!r} ({len(segment)} znakow)"

    print("== sending (sequentially for debugging) ==")
    call_api_one_by_one(actions)

    print("== done ==")
    print(call_api({"action": "done"}))


if __name__ == "__main__":
    main()

import os
import re
import ast
import json
import time
import queue
import operator
import threading
import datetime
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests
from google import genai
from google.genai import types

API_KEY = os.environ.get("AIDEVS_API_KEY")
MODEL = "gemini-3.8-flash"

BASE_URL = "https://hub.ag3nts.org/"
VERIFY_URL = BASE_URL + "verify"
DOCS_URL = BASE_URL + "dane/timetravel.md"
PREVIEW_URL = BASE_URL + "timetravel_preview"
TASK_NAME = "timetravel"

client = genai.Client()

STOP = threading.Event()
PRINT_LOCK = threading.Lock()
INBOX = {"api-agent": queue.Queue(), "ui-agent": queue.Queue()}

FLAG_RE = re.compile(r"\{FLG:[^}]+\}")
SERVER_TOOLS = {"timetravel_api", "get_config", "http_request", "fire_when_ready"}


def log(name, msg):
    """
    Thread-safe logging utility for printing messages with a component prefix.

    Parameters
    ----------
    name : str
        Name of the component or logger.
    msg : str
        Message to print.

    Returns
    -------
    None
    """

    with PRINT_LOCK:
        print(f"[{name}] {msg}")


def _safe_url(url):
    """
    Constructs a full URL and validates that it belongs to the allowed domain.

    Parameters
    ----------
    url : str
        Relative or absolute URL path.

    Returns
    -------
    str
        Validated full URL.

    Raises
    ------
    ValueError
        If the hostname is not hub.ag3nts.org.
    """

    full = urljoin(BASE_URL, url)

    if urlparse(full).hostname != "hub.ag3nts.org":
        raise ValueError("Only the hub.ag3nts.org domain is allowed")

    return full


def timetravel_api(
    action: str, param: Optional[str] = None, value: Optional[float] = None
) -> str:
    """
    Call the time machine API (/verify, task timetravel).

    Parameters
    ----------
    action : str
        Action to perform ('help', 'getConfig', 'reset', or 'configure').
    param : str, optional
        Configuration parameter (day, month, year, syncRatio, stabilization).
    value : float or int, optional
        Value to assign to the configuration parameter.

    Returns
    -------
    str
        JSON-formatted response string from the API or an error message.
    """

    answer = {"action": action}

    if param is not None:
        answer["param"] = param

    if value is not None:
        answer["value"] = (
            int(value) if float(value).is_integer() and param != "syncRatio" else value
        )

    for attempt in range(5):
        try:
            r = requests.post(
                VERIFY_URL,
                json={"apikey": API_KEY, "task": TASK_NAME, "answer": answer},
                timeout=15,
            )
            try:
                return json.dumps(r.json(), ensure_ascii=False)
            except ValueError:
                pass
        except requests.RequestException:
            pass
        time.sleep(1)

    return "Error: API is not responding correctly after 5 attempts."


def get_config() -> str:
    """
    Retrieve the current device state and configuration (getConfig).

    Returns
    -------
    str
        JSON string containing device parameters: date, syncRatio, stabilization,
        fluxDensity, batteryStatus, PTA, PTB, PWR, mode, internalMode, and condition.
    """

    return timetravel_api("getConfig")


def http_request(method: str, url: str, json_body: Optional[str] = None) -> str:
    """
    Perform an HTTP request to hub.ag3nts.org (e.g., web interface endpoints found in page source).

    Parameters
    ----------
    method : str
        HTTP method ('GET' or 'POST').
    url : str
        Full or relative URL.
    json_body : str, optional
        Request body as a JSON string (or omitted).

    Returns
    -------
    str
        Response status code and content preview, or an error message.
    """

    try:
        full = _safe_url(url.replace("{{API_KEY}}", API_KEY))
        body = (
            json.loads(json_body.replace("{{API_KEY}}", API_KEY)) if json_body else None
        )

    except Exception as e:
        return f"Parameter error: {e}"

    for _ in range(3):
        try:
            r = requests.request(method.upper(), full, json=body, timeout=15)
            return f"status={r.status_code}\n{r.text[:3000]}"
        except requests.RequestException:
            time.sleep(1)

    return "Error: no response after 3 attempts."


def wait(seconds: float) -> str:
    """
    Wait for a specified number of seconds (max 30).

    Parameters
    ----------
    seconds : float
        Number of seconds to pause execution.

    Returns
    -------
    str
        Confirmation message with the elapsed wait time.
    """

    seconds = max(0.0, min(float(seconds), 30.0))
    time.sleep(seconds)

    return f"Waited {seconds} s."


def make_messaging(name):
    """
    Create inter-agent messaging functions (`send_message`, `receive_message`)
    for a given agent name.

    Parameters
    ----------
    name : str
        Identifier of the agent ('api-agent' or 'ui-agent').

    Returns
    -------
    tuple of function
        A tuple containing (send_message, receive_message).
    """

    def send_message(to: str, text: str) -> str:
        """
        Send a message to another agent ('api-agent' or 'ui-agent').

        Parameters
        ----------
        to : str
            Recipient agent name.
        text : str
            Message body as plain text or JSON.

        Returns
        -------
        str
            Status message indicating success or unknown recipient.

        """

        if to not in INBOX or to == name:
            return f"Unknown recipient: {to}"

        INBOX[to].put(text)

        return "Sent."

    def receive_message(timeout_s: float = 60) -> str:
        """
        Wait for a message from another agent.

        Parameters
        ----------
        timeout_s : float, optional
            Maximum time to wait in seconds (capped at 120).

        Returns
        -------
        str
            Content of the message, or an indicator if no message arrived or if stopped.
        """

        deadline = time.time() + min(float(timeout_s), 120)

        while time.time() < deadline:
            if STOP.is_set():
                return "STOP: mission completed (flag found)."
            try:
                return INBOX[name].get(timeout=1)
            except queue.Empty:
                pass

        return "NO MESSAGE"

    return send_message, receive_message


def _fetch_docs():
    """
    Fetch the device documentation from the remote server with automatic retries.

    Returns
    -------
    str
        Raw documentation text, or an empty string if retrieval fails.
    """

    for _ in range(5):
        try:
            r = requests.get(DOCS_URL, timeout=15)
            r.raise_for_status()
            return r.text
        except requests.RequestException:
            time.sleep(1)

    return ""


def read_docs(query: Optional[str] = None) -> str:
    """
    Read and filter the device documentation fetched live from the server.

    Parameters
    ----------
    query : str, optional
        Regex pattern (case-insensitive) to filter specific lines or table rows.

    Returns
    -------
    str
        Filtered documentation lines or the full document with large tables omitted.
    """

    text = _fetch_docs()
    if not text:
        return "Failed to download documentation."
    lines = text.splitlines()

    if not query:
        kept = [
            ln
            for ln in lines
            if not (ln.lstrip().startswith("|") and ln.count("|") > 10)
        ]
        kept.append(
            "\n[Large table omitted. Use read_docs(query=...) with a value, e.g., a year.]"
        )
        return "\n".join(kept)

    try:
        pat = re.compile(query, re.I)
    except re.error:
        pat = re.compile(re.escape(query), re.I)

    hits = [ln for ln in lines if pat.search(ln)]

    return "\n".join(hits) if hits else "No matches found."


_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval(node):
    """
    Safely evaluate a simple mathematical AST expression node.

    Parameters
    ----------
    node : ast.AST
        AST node to evaluate.

    Returns
    -------
    int or float
        Result of the evaluated expression.

    Raises
    ------
    ValueError
        If the expression contains disallowed nodes or operations.
    """

    if isinstance(node, ast.Expression):
        return _eval(node.body)

    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value

    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))

    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))

    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "round"
    ):
        return round(*[_eval(a) for a in node.args])

    raise ValueError("Disallowed expression")


def calculate(expression: str) -> str:
    """
    Safely evaluate an arithmetic expression (e.g., '(5*8 + 11*12 +
    2238*7) % 101' or 'round(82/100, 2)').

    Allowed operations: + - * / // % ** and round().

    Parameters
    ----------
    expression : str
        Mathematical expression string to evaluate.

    Returns
    -------
    str
        Evaluated result as a string, or an error message if evaluation
        fails.
    """

    try:
        return str(_eval(ast.parse(expression, mode="eval")))

    except Exception as e:
        return f"Calculation error: {e}"


def fetch_page(url: str, query: Optional[str] = None, context: int = 250) -> str:
    """
    Fetch the raw source code of a page or file (HTML, JS) from
    hub.ag3nts.org (e.g., url='timetravel_preview').

    Without a query, it returns the beginning of the file (6000
    characters) and a list of external scripts.
    With a regex query, it returns snippets of code around matches;
    context defines the number of characters
    before and after each match (default 250, max 2000).

    Parameters
    ----------
    url : str
        Target URL or path.
    query : str, optional
        Regex pattern to search for within the page source.
    context : int, optional
        Number of characters to include around each match.

    Returns
    -------
    str
        Page metadata along with script lists, matching snippets,
        or raw content.
    """

    try:
        r = requests.get(_safe_url(url), timeout=15)
        text = r.text

    except Exception as e:
        return f"Fetch error: {e}"

    if query:
        try:
            pat = re.compile(query, re.I)
        except re.error:
            pat = re.compile(re.escape(query), re.I)

        ctx = max(50, min(int(context), 2000))
        chunks, last_end = [], -1
        for m in pat.finditer(text):
            if m.start() < last_end:
                continue
            s, e = max(0, m.start() - ctx), min(len(text), m.end() + ctx)
            chunks.append(text[s:e].replace("\n", " "))
            last_end = e
            if len(chunks) >= 8:
                break

        return f"[status={r.status_code}, length={len(text)}]\n" + (
            "\n---\n".join(chunks) if chunks else "No matches found."
        )

    scripts = re.findall(r'<script[^>]+src=["\']([^"\']+)', text, re.I)

    return f"[status={r.status_code}, length={len(text)}]\nExternal scripts: {scripts}\n\n{text[:6000]}"


def fire_when_ready(
    required_internal_mode: int,
    method: str,
    url: str,
    json_body: Optional[str] = None,
    timeout_s: float = 90,
) -> str:
    """
    Wait until the device is ready for a jump (mode='active',
    fluxDensity=100, and internalMode matching required_internal_mode)
    and INSTANTLY execute the provided HTTP request (e.g., triggering the
    jump sphere).

    Polls the state every 0.3 seconds to catch short transition windows.

    Parameters
    ----------
    required_internal_mode : int
        Target internal mode required for the jump.
    method : str
        HTTP method for the trigger request ('GET' or 'POST').
    url : str
        Target URL endpoint.
    json_body : str, optional
        JSON payload string for the request.
    timeout_s : float, optional
        Maximum timeout in seconds (capped at 180).

    Returns
    -------
    str
        Response of the triggered request upon readiness, or the final
        state upon timeout.
    """

    deadline = time.time() + min(float(timeout_s), 180)
    last = {}

    while time.time() < deadline and not STOP.is_set():
        try:
            last = json.loads(timetravel_api("getConfig")).get("config", {})
        except Exception:
            time.sleep(0.3)
            continue

        if (
            last.get("mode") == "active"
            and last.get("fluxDensity", 0) >= 100
            and last.get("internalMode") == int(required_internal_mode)
        ):
            return "Request sent upon readiness:\n" + http_request(
                method, url, json_body
            )
        time.sleep(0.3)

    return f"Timeout. Last state: {json.dumps(last, ensure_ascii=False)}"


def run_agent(name, system, mission, tool_funcs, max_steps=150):
    """
    Run an autonomous agent loop using Google GenAI models with tool
    calling support.

    Parameters
    ----------
    name : str
        Identifier name of the agent.
    system : str
        System instructions for the model.
    mission : str
        Initial mission statement or user prompt.
    tool_funcs : list of callable
        List of Python functions available as tools for the agent.
    max_steps : int, optional
        Maximum number of interaction steps before stopping (default 150).

    Returns
    -------
    None
    """

    tools = {f.__name__: f for f in tool_funcs}
    config = types.GenerateContentConfig(
        system_instruction=system,
        tools=list(tools.values()),
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    contents = [types.Content(role="user", parts=[types.Part(text=mission)])]
    nudges = 0

    for _ in range(max_steps):
        if STOP.is_set():
            break

        try:
            resp = client.models.generate_content(
                model=MODEL, contents=contents, config=config
            )
        except Exception as e:
            log(name, f"[!] Model error: {e}, retrying")
            time.sleep(2)
            continue

        if not resp.candidates or not resp.candidates[0].content:
            time.sleep(1)
            continue

        content = resp.candidates[0].content
        contents.append(content)
        parts = content.parts or []

        text = " ".join(p.text.strip() for p in parts if p.text)
        if text:
            log(name, f"🤖 {text}")

        calls = [p.function_call for p in parts if p.function_call]

        if not calls:
            if "[KONIEC]" in text or nudges >= 3:
                break
            nudges += 1
            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            text="Misja nie jest jeszcze zakończona. Użyj narzędzi (np. receive_message) i kontynuuj. "
                            "Gdy naprawdę skończysz, napisz [KONIEC]."
                        )
                    ],
                )
            )
            continue
        nudges = 0

        results = []
        for fc in calls:
            args = dict(fc.args or {})
            log(name, f"🔧 {fc.name}({json.dumps(args, ensure_ascii=False)[:220]})")

            try:
                out = tools[fc.name](**args)
            except Exception as e:
                out = f"Tool error: {e}"

            log(name, f"   ↳ {str(out)[:350]}")
            flag = FLAG_RE.search(str(out)) if fc.name in SERVER_TOOLS else None
            if flag:
                log(name, f"🎉 FLAG: {flag.group(0)}")
                log(name, f"Full response: {out}")
                STOP.set()
            results.append(
                types.Part.from_function_response(
                    name=fc.name, response={"result": out}
                )
            )

        contents.append(types.Content(role="user", parts=results))

    log(name, "Agent execution finished.")


API_SYSTEM = """Jesteś api-agent: agent-planista i operator BACKENDU maszyny czasu CHRONOS-P1. Współpracujesz
z ui-agent, który obsługuje interfejs WWW (PT-A, PT-B, PWR, standby/active, klik sfery skoku).
Komunikujecie się wyłącznie przez send_message / receive_message. Działasz w pełni autonomicznie.

Zasady:
1. Zacznij od przeczytania dokumentacji (read_docs). Wzory, zakresy, tabele i ograniczenia bierz wyłącznie
   z niej, nigdy z pamięci. Dokumentacja może się zmienić między uruchomieniami.
2. NIGDY nie licz w pamięci: każde obliczenie rób narzędziem calculate, wprost wg wzoru z dokumentacji.
   Wartości zależne od roku (np. ochrona PWR) czytaj przez read_docs z query (np. rok).
3. Sam zaplanuj etapy misji i dla każdego ustal: datę, kierunek (który z PT-A/PT-B; tryb tunelu),
   PWR, wymagany internalMode, stan baterii.
4. Konfigurację przez API robisz TYLKO w trybie standby (sprawdź get_config). Kolejność: pełna data
   (year, month, day), syncRatio, potem stabilization wg podpowiedzi API (np. pole needConfig w odpowiedzi;
   nie zgaduj). Sprawdź wynik w get_config.
5. Gdy Twoja część etapu jest gotowa, wyślij do ui-agent wiadomość-zadanie (JSON) z polami: stage, opis,
   target_date, PT_A (true/false), PT_B (true/false), PWR, required_internal_mode oraz polecenie:
   ustaw PT/PWR, przełącz na active, w odpowiednim momencie wykonaj skok/otwórz tunel i zaraportuj.
   Następnie czekaj na raport (receive_message) i sam zweryfikuj skutek w get_config (data, bateria, tryb).
6. Pilnuj baterii (batteryStatus), nie rozładuj jej do zera. Przy błędach analizuj odpowiedzi i popraw
   przyczynę. NIGDY nie używaj reset: kasuje postęp (data wraca do dzisiejszej, bateria do 1/3).
7. Gdy wszystkie etapy są wykonane (albo dostaniesz flagę), wyślij do ui-agent wiadomość "done",
   napisz krótkie podsumowanie po polsku i zakończ tekstem [KONIEC]."""

UI_SYSTEM = f"""Jesteś ui-agent: agent obsługujący FRONTEND maszyny czasu CHRONOS-P1 ({PREVIEW_URL}).
Współpracujesz z api-agent, który konfiguruje datę/syncRatio/stabilization przez API i planuje etapy.
Komunikujecie się przez send_message / receive_message. Działasz w pełni autonomicznie.

Zasady:
1. Na starcie poznaj interfejs: przez fetch_page przeczytaj kod strony i ustal, jakie żądania HTTP wysyła
   (ustawianie PTA/PTB/PWR/mode oraz klik sfery aktywującej skok/tunel), z jakim URL-em, metodą i ciałem.
   Szukaj m.in. wzorców: fetch(, /timetravel_backend, /verify, sendUpdate, orb. Używaj parametru context,
   żeby zobaczyć całe wywołania z ciałem. Klucz API podawaj jako {{{{API_KEY}}}}.
2. Potem w pętli: receive_message -> wykonaj zadanie -> raport. Zadanie polega na ustawieniu PT-A/PT-B/PWR
   i mode='active' przez znaleziony endpoint, zweryfikowaniu tego w get_config, a następnie wykonaniu skoku
   lub otwarcia tunelu narzędziem fire_when_ready, które czeka na fluxDensity=100 i właściwy internalMode
   i wysyła kliknięcie sfery bez opóźnienia (nie próbuj łapać okna ręcznie).
3. Nie zmieniaj date/syncRatio/stabilization (to robi api-agent) i nie używaj reset.
4. Po skoku sprawdź get_config i wyślij api-agent raport: co zrobiłeś, pełny stan (data, bateria, mode,
   PTA/PTB/PWR) oraz ewentualny komunikat lub flagę. Jeśli coś się nie udało, opisz dokładnie dlaczego.
5. Gdy dostaniesz "done" albo STOP, napisz krótkie podsumowanie i zakończ tekstem [KONIEC]."""

MISSION = f"""Dzisiejsza data: {datetime.date.today().isoformat()}.
CEL: otworzyć TUNEL czasowy do 12 listopada 2024 (dzień przed tym, jak Rafał został znaleziony w jaskini).
Nie mamy dość energii na tunel, więc plan zakłada jeden dodatkowy skok:
1. Skok do 5 listopada 2238, gdzie nasz człowiek wręczy nową paczkę baterii.
2. Po wymianie baterii powrót do teraźniejszości (dzisiejsza data).
3. Z teraźniejszości otwarcie tunelu do 12 listopada 2024.
Zadanie kończy się sukcesem, gdy tunel zostanie otwarty i pojawi się flaga (FLG).

WAŻNE: na starcie sprawdź stan urządzenia (getConfig). Pole currentDate mówi, w jakim czasie jesteś
(dzisiejsza data = teraźniejszość), a bateria wraca do 1/3 dopiero po resecie. Jeśli skok do 2238
i powrót są już wykonane (jesteś w teraźniejszości z baterią co najmniej 2/3), zaplanuj tylko
pozostałe etapy i nie powtarzaj wykonanych."""


def main():
    """
    Initialize and run the multi-agent system consisting of an API agent
    and a UI agent.

    Returns
    -------
    None

    Raises
    ------
    SystemExit
        If the AIDEVS_API_KEY environment variable is not set.
    """

    if not API_KEY:
        raise SystemExit("AIDEVS_API_KEY environment variable is missing")

    api_send, api_recv = make_messaging("api-agent")
    ui_send, ui_recv = make_messaging("ui-agent")

    threads = [
        threading.Thread(
            target=run_agent,
            args=(
                "api-agent",
                API_SYSTEM,
                MISSION
                + "\n\nTy jesteś api-agent: zaplanuj misję i prowadź ją etap po etapie.",
                [read_docs, calculate, timetravel_api, wait, api_send, api_recv],
            ),
        ),
        threading.Thread(
            target=run_agent,
            args=(
                "ui-agent",
                UI_SYSTEM,
                MISSION
                + "\n\nTy jesteś ui-agent: poznaj interfejs, a potem czekaj na zadania od api-agent.",
                [
                    fetch_page,
                    http_request,
                    get_config,
                    fire_when_ready,
                    wait,
                    ui_send,
                    ui_recv,
                ],
            ),
        ),
    ]
    print("=== Two agents: api-agent + ui-agent ===")
    for t in threads:
        t.start()
    for t in threads:
        t.join()


if __name__ == "__main__":
    main()

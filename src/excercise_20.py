import requests
import json
import os
from google import genai

API_KEY = os.environ.get("AIDEVS_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
VERIFY_URL = "https://hub.ag3nts.org/verify"
FOOD4CITIES_URL = "https://hub.ag3nts.org/dane/food4cities.json"
TASK = "foodwarehouse"

client = genai.Client()


def call_api(answer: dict):
    """
    Verifies answer with api with hub

    Parameters
    ----------
    answer: dict
        Answer to be uploaded

    Returns
    -------
    Json file with status after sending the answer
    """

    payload = {"apikey": API_KEY, "task": TASK, "answer": answer}
    r = requests.post(VERIFY_URL, json=payload, timeout=30)
    r.raise_for_status()

    return r.json()


def generate_title(city_name: str) -> str:
    """
    Generate short title of order using Gemini.

    Parameters
    ----------
    city_name: str
        Name of the city for which the title should be generated

    Returns
    -------
    String containing generated title of the order
    """

    try:
        prompt = (
            f"Wygeneruj krótki (max 6 słów), rzeczowy tytuł zamówienia "
            f"magazynowego dla dostawy do miasta {city_name.capitalize()}. "
            f"Zwróć tylko sam tytuł, bez cudzysłowów, bez dodatkowego tekstu."
        )
        resp = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt,
        )
        title = resp.text.strip().strip('"')
        return title if title else f"Delivery to {city_name.capitalize()}"
    except Exception as e:
        print(f"Gemini fallback for {city_name}: {e}")
        return f"Delivery to {city_name.capitalize()}"


# --- 0. Reset of status ---
reset_resp = call_api({"tool": "reset"})
print("Reset:", reset_resp["message"])

# --- 1. Cities needs ---
cities_data = requests.get(FOOD4CITIES_URL, timeout=30).json()
city_names = list(cities_data.keys())
print("Cities:", city_names)

# --- 2. Destinations only for required cities ---
capitalized_names = [c.capitalize() for c in city_names]
in_clause = ", ".join(f"'{name}'" for name in capitalized_names)
dest_query = f"select * from destinations where name in ({in_clause})"
destinations_resp = call_api({"tool": "database", "query": dest_query})
print("destinations:", json.dumps(destinations_resp, ensure_ascii=False, indent=2))

destinations_rows = destinations_resp["rows"]
dest_by_city = {
    row["name"].strip().lower(): row["destination_id"] for row in destinations_rows
}

missing = [c for c in city_names if c.lower() not in dest_by_city]
if missing:
    raise RuntimeError(
        f"Missing destination_id for cities: {missing} — check spelling in database."
    )

# --- 3. Checking roles (to confirm, that role==2 can create orders) ---
roles_resp = call_api({"tool": "database", "query": "select * from roles"})
print("roles:", json.dumps(roles_resp, ensure_ascii=False, indent=2))

# --- 4. Selecting creator: active user with role == 2 ---
users_resp = call_api(
    {
        "tool": "database",
        "query": "select * from users where role = 2 and is_active = 1",
    }
)
print("users (role=2):", json.dumps(users_resp, ensure_ascii=False, indent=2))

users_rows = users_resp["rows"]
if not users_rows:
    raise RuntimeError(
        "No active users with role=2 - verify statements about the role in database."
    )

creator = users_rows[0]
creator_id = creator["user_id"]
creator_login = creator["login"]
creator_birthday = creator["birthday"]
print(f"Chosen creator: {creator_login} (id={creator_id}, birthday={creator_birthday})")

# --- 5. Creating orders for each city ---
created_orders = []

for city_name, items in cities_data.items():
    dest_code = dest_by_city[city_name.lower()]

    sig_resp = call_api(
        {
            "tool": "signatureGenerator",
            "action": "generate",
            "login": creator_login,
            "birthday": creator_birthday,
            "destination": dest_code,
        }
    )
    signature = sig_resp.get("hash")
    if not signature:
        raise RuntimeError(f"Brak signature w odpowiedzi dla {city_name}: {sig_resp}")
    print(f"{city_name}: signature = {signature}")

    title = generate_title(city_name)

    create_resp = call_api(
        {
            "tool": "orders",
            "action": "create",
            "title": title,
            "creatorID": creator_id,
            "destination": dest_code,
            "signature": signature,
        }
    )
    order = create_resp.get("order", create_resp)
    order_id = order.get("id")
    if not order_id:
        raise RuntimeError(
            f"Error during creating order for {city_name}: {create_resp}"
        )
    print(f"{city_name}: order_id = {order_id}, title = '{title}'")

    append_resp = call_api(
        {"tool": "orders", "action": "append", "id": order_id, "items": items}
    )
    print(f"{city_name}: append_resp = {append_resp}")

    created_orders.append(order_id)

print(
    f"\nCreated {len(created_orders)} orders from required cities: {len(city_names)}."
)

# --- 6. Final verification ---
done_resp = call_api({"tool": "done"})
print("DONE:", json.dumps(done_resp, ensure_ascii=False, indent=2))

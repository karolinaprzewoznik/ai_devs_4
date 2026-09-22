import os
import json
import time
import requests

CENTRAL_URL = "https://hub.ag3nts.org/verify"
API_KEY = os.environ["AIDEVS_API_KEY"]
TASK = "windpower"

session = requests.Session()


# ============================================================
# API
# ============================================================


def call(action, **kwargs):
    payload = {
        "apikey": API_KEY,
        "task": TASK,
        "answer": {
            "action": action,
            **kwargs,
        },
    }

    r = session.post(
        CENTRAL_URL,
        json=payload,
        timeout=4,
    )

    try:
        data = r.json()
    except Exception:
        data = {"raw": r.text}

    print(
        f">>> {action} "
        f"{kwargs if kwargs else ''} "
        f"=> {r.status_code} "
        f"{json.dumps(data, ensure_ascii=False)}"
    )

    return data


# ============================================================
# START
# ============================================================

start = call("start")

if start.get("code") != 60:
    raise RuntimeError(f"START ERROR: {start}")

print("SESSION:", start.get("sessionStart"))


# ============================================================
# ASYNCHRONIC DATA COLLECTION
# ============================================================

call("get", param="weather")
call("get", param="powerplantcheck")
call("get", param="turbinecheck")


# ============================================================
# RAW DATA COLLECTION
# ============================================================

weather = None
powerplant = None
turbine = None

deadline = time.monotonic() + 32

while time.monotonic() < deadline:

    result = call("getResult")

    if result.get("code") != 12:
        time.sleep(0.08)
        continue

    source = result.get("sourceFunction")

    if source == "weather":
        weather = result
        print("\n### WEATHER RECEIVED ###")

    elif source == "powerplantcheck":
        powerplant = result
        print("\n### POWERPLANT RECEIVED ###")
        print(json.dumps(result, indent=2, ensure_ascii=False))

    elif source == "turbinecheck":
        turbine = result
        print("\n### TURBINE RECEIVED ###")
        print(json.dumps(result, indent=2, ensure_ascii=False))

    if weather and powerplant and turbine:
        break


if weather is None:
    raise RuntimeError(
        "Error in weather data download:\n"
        + json.dumps(weather, indent=2, ensure_ascii=False)
    )

if powerplant is None:
    raise RuntimeError(
        "Error in powerplantcheck data download:\n"
        + json.dumps(powerplant, indent=2, ensure_ascii=False)
    )

if turbine is None:
    raise RuntimeError(
        "Error in turbinecheck data download:\n"
        + json.dumps(turbine, indent=2, ensure_ascii=False)
    )


# ============================================================
# WEATHER FORECAST
# ============================================================


def find_forecast(obj):
    """
    Finds the list of forecast records regardless of whether
    the API wraps it in forecast/data/weather, or any other structure.
    """

    if isinstance(obj, list):
        if obj and all(isinstance(x, dict) for x in obj):
            return obj

        for item in obj:
            found = find_forecast(item)
            if found:
                return found

    if isinstance(obj, dict):

        for key in (
            "forecast",
            "weather",
            "data",
            "records",
            "points",
        ):
            if key in obj:
                found = find_forecast(obj[key])
                if found:
                    return found

        for value in obj.values():
            found = find_forecast(value)
            if found:
                return found

    return None


records = find_forecast(weather)

if not records:
    raise RuntimeError(
        "Error in weather data download:\n"
        + json.dumps(weather, indent=2, ensure_ascii=False)
    )


# ============================================================
# RECORDS CLEANUP
# ============================================================


def get(record, *names):
    for name in names:
        if name in record:
            return record[name]
    return None


forecast = []

for record in records:

    dt = get(
        record,
        "datetime",
        "timestamp",
        "time",
        "hour",
        "startHour",
    )

    wind = get(
        record,
        "windMs",
        "wind",
        "windSpeed",
        "windSpeedMs",
    )

    if dt is None or wind is None:
        continue

    try:
        wind = float(wind)
    except (TypeError, ValueError):
        continue

    forecast.append(
        {
            "datetime": str(dt),
            "windMs": wind,
        }
    )


if not forecast:
    raise RuntimeError(
        "Error in weather data download:\n"
        + json.dumps(weather, indent=2, ensure_ascii=False)
    )


# ============================================================
# POWER CURVE FROM DOCUMENTATION
#
# Rated power = 14 kW
#
# 4 m/s  -> 10-15%
# 6 m/s  -> 30-40%
# 8 m/s  -> 60-70%
# 10 m/s -> 90-100%
#
# We use the upper bound of the intervals to check if a point can
# provide the required 2-3 kW.
#
# For 4.9 m/s:
#
# 15% + (4.9-4)/(6-4) * (30%-15%)
# = 21.75%
#
# 14 kW * 21.75% = 3.045 kW
#
# enough for deficit 2-3 kW.
# ============================================================

RATED_POWER_KW = 14.0

POWER_CURVE = [
    (4.0, 0.15),
    (6.0, 0.40),
    (8.0, 0.70),
    (10.0, 1.00),
    (12.0, 1.00),
    (14.0, 1.00),
]


def estimated_power_kw(wind):
    """
    Estimate power from the upper bounds of the documentation intervals.
    """

    if wind < 4:
        return 0.0

    if wind >= 14:
        return 0.0

    for i in range(len(POWER_CURVE) - 1):

        v1, p1 = POWER_CURVE[i]
        v2, p2 = POWER_CURVE[i + 1]

        if v1 <= wind <= v2:

            fraction = (wind - v1) / (v2 - v1)

            percent = p1 + fraction * (p2 - p1)

            return RATED_POWER_KW * percent

    return 0.0


# ============================================================
# DEFICIT PARSING
# ============================================================


def parse_deficit(value):
    """
    '2-3' -> 3.0
    '3'   -> 3.0
    """

    text = str(value).strip()

    if "-" in text:
        parts = text.split("-")

        try:
            return float(parts[-1])
        except ValueError:
            pass

    try:
        return float(text)
    except ValueError:
        raise RuntimeError(f"Unknown powerDeficitKw: {value}")


required_power = parse_deficit(powerplant.get("powerDeficitKw", "3"))

print("\nREQUIRED POWER:", required_power, "kW")


# ============================================================
# SORT FORECAST
# ============================================================

forecast.sort(key=lambda x: x["datetime"])


# ============================================================
# STORM POINTS
#
# All points > 14 m/s.
#
# pitch = 90
# turbineMode = idle
# ============================================================

storm_points = [point for point in forecast if point["windMs"] > 14]


# ============================================================
# PRODUCTION POINT
#
# First point that:
#
#   - is not a storm
#   - has >= 4 m/s
#   - provides the required power
#
# pitch = 0
# turbineMode = production
# ============================================================

production_point = None

for point in forecast:

    wind = point["windMs"]

    if wind >= 14:
        continue

    power = estimated_power_kw(wind)

    if power >= required_power:
        production_point = point.copy()
        production_point["estimatedPowerKw"] = power
        break


if production_point is None:
    raise RuntimeError(
        "Error in weather data download:\n"
        + json.dumps(forecast, indent=2, ensure_ascii=False)
    )


# ============================================================
# CONFIG BUILDER
# ============================================================

configs = []

seen = set()


# --- STORM POINTS ---

for point in storm_points:

    key = point["datetime"]

    if key in seen:
        continue

    seen.add(key)

    configs.append(
        {
            "datetime": key,
            "windMs": point["windMs"],
            "pitchAngle": 90,
            "turbineMode": "idle",
            "reason": "storm",
        }
    )


# --- PRODUCTION POINT ---

key = production_point["datetime"]

if key in seen:
    raise RuntimeError("Production point collides with a storm point.")

configs.append(
    {
        "datetime": key,
        "windMs": production_point["windMs"],
        "pitchAngle": 0,
        "turbineMode": "production",
        "reason": "production",
    }
)


# ============================================================
# SORTING
# ============================================================

configs.sort(key=lambda x: x["datetime"])


print("\n" + "=" * 70)
print("PLAN")
print("=" * 70)

for c in configs:

    print(
        c["datetime"],
        "| wind =",
        c["windMs"],
        "| pitch =",
        c["pitchAngle"],
        "| mode =",
        c["turbineMode"],
        "| reason =",
        c["reason"],
        (f"| estimatedPower = " f"{c.get('estimatedPowerKw', 0):.3f} kW"),
    )


# ============================================================
# UNLOCK CODE GENERATOR
# ============================================================

print("\n" + "=" * 70)
print("QUEUE UNLOCK CODES")
print("=" * 70)

for c in configs:

    date, hour = c["datetime"].split(" ", 1)

    call(
        "unlockCodeGenerator",
        startDate=date,
        startHour=hour,
        windMs=c["windMs"],
        pitchAngle=c["pitchAngle"],
    )


# ============================================================
# UNLOCK CODES
# ============================================================

unlock_codes = {}

deadline = time.monotonic() + 30

while time.monotonic() < deadline:

    result = call("getResult")

    if result.get("code") != 12:
        time.sleep(0.08)
        continue

    if result.get("sourceFunction") != "unlockCodeGenerator":
        continue

    signed = result.get("signedParams", {})

    date = signed.get("startDate")
    hour = signed.get("startHour")
    code = result.get("unlockCode")

    if date and hour and code:

        key = f"{date} {hour}"

        unlock_codes[key] = code

        print(
            "UNLOCK:",
            key,
            "wind=",
            signed.get("windMs"),
            "pitch=",
            signed.get("pitchAngle"),
            "code=",
            code,
        )

    if len(unlock_codes) == len(configs):
        break


# ============================================================
# VALIDATION
# ============================================================

expected = {c["datetime"] for c in configs}

missing = expected - set(unlock_codes)

if missing:
    raise RuntimeError("Missing unlockCode for:\n" + "\n".join(sorted(missing)))


# ============================================================
# FINAL CONFIG
# ============================================================

batch = {}

for c in configs:

    key = c["datetime"]

    batch[key] = {
        "pitchAngle": c["pitchAngle"],
        "turbineMode": c["turbineMode"],
        "unlockCode": unlock_codes[key],
    }


print("\n" + "=" * 70)
print("FINAL CONFIG")
print("=" * 70)

print(
    json.dumps(
        batch,
        indent=2,
        ensure_ascii=False,
    )
)


# ============================================================
# CONFIG
# ============================================================

config_result = call(
    "config",
    configs=batch,
)

if config_result.get("code") != 10:
    raise RuntimeError(f"CONFIG ERROR: {config_result}")


# ============================================================
# DONE
# ============================================================

done_result = call("done")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

print(
    json.dumps(
        done_result,
        indent=2,
        ensure_ascii=False,
    )
)

if done_result.get("code") != 200:
    print("\nDONE NOT COMPLETED.")
else:
    print("\nSUCCESS.")

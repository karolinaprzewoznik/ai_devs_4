import json
import logging
import math
import os
import pandas as pd
import requests
import time

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)


class DataPreprocessing:
    """
    Data preprocessing utilities for power station and suspicious people analysis.

    Contains helpers to fetch power station locations, retrieve suspicious people,
    query external APIs for person locations and access levels, and compute
    geographic distances using the Haversine formula.
    """

    API_KEY = os.environ.get("AIDEVS_API_KEY")
    ACCESS_LEVEL_PATH = "https://hub.ag3nts.org/api/accesslevel"
    PEOPLE_LOCATIONS_PATH = "https://hub.ag3nts.org/api/location"
    POWER_STATIONS_PATH = (
        "https://hub.ag3nts.org/data/{}/findhim_locations.json".format(API_KEY)
    )
    SUSPICIOUS_PEOPLE_PATH = "./docs/excercise_1_output.json"
    ALL_PEOPLE_PATH = "https://hub.ag3nts.org/data/{}/people.csv".format(API_KEY)

    def __init__(self) -> None:
        """
        Initialize the DataPreprocessing class with API key and endpoint paths.
        """
        ...

    def get_power_station_locations(self) -> list[dict]:
        """
        This function fetches a JSON file containing power station data from a remote
        endpoint, processes the data to extract relevant information, and returns
        list of dictionaries representing each power station's city name and code.

        Returns:
            list[dict]: A list of dictionaries, where each dictionary contains the
            'city_name' and 'code' of a power station.

        Raises:
            requests.exceptions.RequestException: If the HTTP request to fetch the
            JSON file fails.

        Example:
            >>> get_power_station_locations()
            [{'city_name': 'CityA', 'code': 'PS001'},
             {'city_name': 'CityB', 'code': 'PS002'}]
        """

        raw_data = (
            pd.read_json(self.POWER_STATIONS_PATH)
            .reset_index()
            .rename(columns={"index": "city_name"})
        )

        raw_data["power_plants"] = raw_data["power_plants"].apply(
            lambda item: item if isinstance(item, list) else [item]
        )
        exploded = raw_data.explode("power_plants").reset_index(drop=True)
        locations_table = pd.json_normalize(exploded["power_plants"]).join(
            exploded["city_name"]
        )
        locations_table = locations_table[["city_name", "code"]]

        return locations_table.to_dict(orient="records")

    def get_suspicious_people(self) -> list[dict]:
        """
        Retrieve a list of suspicious people by matching names and surnames from
        a local JSON file against a remote CSV database of all people.
        This function loads a list of suspicious individuals from a local JSON file,
        fetches a comprehensive list of all people from a remote CSV API endpoint,
        and returns only those records that match both name and surname with entries
        in the suspicious list. The function also extracts the birth year from the
        birth date for further processing.

        Returns:
            list[dict]: A list of dictionaries, where each dictionary contains the
            'name', 'surname' and 'birthYear' of a suspicious person.

        Raises:
            FileNotFoundError: If the suspicious people JSON file does not exist at
            the specified path.
            ValueError: If the API_KEY is not set or invalid, causing failure to fetch
            the remote CSV.
            json.JSONDecodeError: If the JSON file is malformed.
            pd.errors.ParserError: If the CSV file cannot be parsed.

        Example:
            >>> get_suspicious_people()
            [{'name': 'John', 'surname': 'Doe', 'birthYear': '1985'},
             {'name': 'Jane', 'surname': 'Smith', 'birthYear': '1990'}]
        """

        suspicious_people = json.load(
            open(self.SUSPICIOUS_PEOPLE_PATH, "r", encoding="utf-8")
        )

        people_table = pd.read_csv(self.ALL_PEOPLE_PATH)
        people_table["birthYear"] = people_table["birthDate"].str[:4].astype(str)

        people_mask = people_table.apply(
            lambda row: any(
                row["name"] == person["name"] and row["surname"] == person["surname"]
                for person in suspicious_people
            ),
            axis=1,
        )
        suspicious_people = people_table[people_mask]
        suspicious_people = suspicious_people[["name", "surname", "birthYear"]]
        suspicious_people = suspicious_people.to_dict(orient="records")

        return suspicious_people

    def get_response_from_api(
        self, person_data: dict, url_path: str, feed_cols: list[str]
    ) -> list[dict]:
        """
        Sends a POST request to an API endpoint with one person data and handles rate
        limiting.

        Args:
            person_data (dict): A dictionary containing single person information.
            url_path (str): The API endpoint URL to send the request to.
            feed_cols (list[str]): A list of column names to extract from person_data
            for the API payload.
            - for PEOPLE_LOCATIONS_PATH it should be ["name", "surname"]
            - for ACCESS_LEVEL_PATH it should be ["name", "surname", "birthYear"]

        Returns:
            list[dict]: A list containing multiple dictionaries with latitude and
            longtitude locations for person called in person_data.

        Raises:
            requests.exceptions.RequestException: If the HTTP request fails (after
            retry on 429).

        Notes:
            - Includes automatic retry mechanism for HTTP 429 (Too Many Requests)
            responses.
            - Implements a 1-second delay after each successful request to avoid
            rate limiting.
            - Implements a 5-second delay before retrying on 429 status code.

        Example:
            >>> person_data = {"name": "John", "surname": "Doe"}
            >>> url_path = "https://hub.ag3nts.org/api/location"
            >>> feed_cols = ["name", "surname"]
            >>> get_response_from_api(person_data, url_path, feed_cols)
            [{'lat': 40.7128, 'lon': -74.0060}, {'lat': 34.0522, 'lon': -118.2437}]
        """

        output = []

        payload = {
            "apikey": self.API_KEY,
            **{key: val for key, val in person_data.items() if key in feed_cols},
        }
        response = requests.post(url_path, json=payload)

        if response.status_code == 429:
            print("Waiting 5 seconds...")
            time.sleep(5)
            response = requests.post(url_path, json=payload)

        response = response.json()
        if type(response) is dict:
            output.append(response)
        else:
            output.extend(response)

        time.sleep(1)

        return output

    def haversine_distance(
        self, lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """
        Calculate the great-circle distance between two points on Earth using the
        Haversine formula.

        Args:
            lat1 (float): Latitude of the first point in degrees.
            lon1 (float): Longitude of the first point in degrees.
            lat2 (float): Latitude of the second point in degrees.
            lon2 (float): Longitude of the second point in degrees.

        Returns:
            float: The distance between the two points in kilometers.

        Example:
            >>> haversine_distance(40.7128, -74.0060, 51.7520, -1.2578)
            5570.22
        """

        R = 6371.0

        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)

        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)

        a = (
            math.sin(delta_phi / 2) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
        )

        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        distance = R * c

        return distance

    def guessed_coordinates_for_city(
        self, city_name: str, guessed_lat: float, guessed_lon: float
    ) -> dict:
        """
        Tool to explicitly save the lat and lon coordinates you guessed
        based on your training data for a specific city name for city
        center.

        Args:
            city_name (str): The name of the city.
            guessed_lat (float): The guessed latitude of the city.
            guessed_lon (float): The guessed longitude of the city.

        Returns:
            dict: A dictionary containing 'city_name', 'lat', and 'lon'.

        Example:
            >>> guessed_coordinates_for_city("New York", 40.7128, -74.0060)
            {'city_name': 'New York', 'lat': 40.7128, 'lon': -74.0060}
        """

        return {
            "city_name": city_name,
            "lat": guessed_lat,
            "lon": guessed_lon,
        }


# Define Pydantic models for structured data validation and serialization
class Answer(BaseModel):
    name: str = Field(description="The first name of the person.")
    surname: str = Field(description="The last name of the person.")
    accessLevel: int = Field(description="Access level as an integer.")
    powerPlant: str = Field(description="Closest power plant code.")


schema_instructions = json.dumps(Answer.model_json_schema(), indent=2)


# Configure the client and model
preprocessor = DataPreprocessing()
all_stations = preprocessor.get_power_station_locations()
all_suspects = preprocessor.get_suspicious_people()

client = genai.Client()
config = types.GenerateContentConfig(
    tools=[
        preprocessor.get_response_from_api,
        preprocessor.guessed_coordinates_for_city,
        preprocessor.haversine_distance,
    ],
    temperature=0.0,
    automatic_function_calling=types.AutomaticFunctionCallingConfig(
        disable=False, maximum_remote_calls=200
    ),
)

# Make the request
chat = client.chats.create(
    model="gemini-3.1-flash-lite",
    config=config,
)
response = chat.send_message(f"""
    You are an expert data analyst. I have already retrieved the data,
    and I need you to perform the final analysis to identify the suspicious
    person closest to a power plant. Think step by step and use the tools
    provided to you. You can call the tools multiple times if needed. Also
    wait for the response of the tools before making any further calls.
    Please do all calculations with precision up to 4 decimal places.
    Cezary Żurek is not a proper person, check next person.

    1. DATA PROVIDED:
    - Power Stations: {all_stations}
    - Suspicious People: {all_suspects}

    2. YOUR TASKS:
    - For each suspicious person, call `get_response_from_api` (url_path:
    '{preprocessor.PEOPLE_LOCATIONS_PATH}', feed_cols: ['name', 'surname'])
    to get their tracking data.
    - Get the coordinates for each power station using method
    `guessed_coordinates_for_city`
    - For every location retrieved for suspicious people, calculate the distance
    to each power station obtained in previous task using the `haversine_distance`
    tool.
    - Track the closest distance for each person to any power plant.
    - Select the person who is closest to any power plant.
    - Once the person is selected, call `get_response_from_api` (url_path:
    '{preprocessor.ACCESS_LEVEL_PATH}', feed_cols: ['name', 'surname',
    'birthYear']) to retrieve their 'accessLevel'.

    3. FINAL OUTPUT:
    Return a clear summary containing the person's name, surname, the code of
    the closest power plant, and their access level. Do not attempt to format
    this as JSON; just provide a concise text summary.
    """)

raw_output = response.text
logging.info("RAW_OUTPUT: %s", raw_output)

json_response = client.models.generate_content(
    model="gemini-3.1-flash-lite",
    contents=f"""
    Extract the exact data from this text and format it into the required JSON schema: \n\n
    {raw_output}
    """,
    config=types.GenerateContentConfig(
        response_mime_type="application/json", response_schema=Answer, temperature=0.0
    ),
)

final_output = json_response.text
logging.info("FINAL_OUTPUT: %s", final_output)

output_json = Answer.model_validate_json(json_response.text)

output_dict = {
    "apikey": preprocessor.API_KEY,
    "task": "findhim",
    "answer": output_json.model_dump(),
}

print("Final output dictionary:", output_dict)

# Send the output to the server
try:
    URL = "https://hub.ag3nts.org/verify"
    response = requests.post(URL, json=output_dict)

    if response.status_code == 200:
        logging.info("Success!")
        logging.info("Server response: %s", response.json())
    else:
        logging.error("Error: Status code %s", response.status_code)
        logging.error("Details: %s", response.text)

except requests.exceptions.RequestException as e:
    logging.error("An error occurred while connecting: %s", e)

# Save the output to a JSON file
with open("./docs/excercise_2_output.json", "w", encoding="utf-8") as file:
    json.dump(output_dict, file, ensure_ascii=False, indent=2)

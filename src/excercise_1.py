import csv
import datetime
import json
import os
import pandas as pd
import requests
from dateutil.relativedelta import relativedelta
from google import genai
from pydantic import BaseModel, Field
from typing import List, Optional


API_KEY = os.environ.get("AIDEVS_API_KEY")

data_path = "https://hub.ag3nts.org/data/{}/people.csv".format(API_KEY)

current_year = datetime.datetime.now().year
min_birth_year = current_year - 20
max_birth_year = current_year - 40

data_table = pd.read_csv(data_path)
data_table["birthYear"] = data_table["birthDate"].str[:4].astype(int)
data_table = data_table[
    (data_table["birthYear"] <= min_birth_year)
    & (data_table["birthYear"] >= max_birth_year)
]
data_table = data_table[
    (data_table["gender"] == "M") & (data_table["birthPlace"] == "Grudziądz")
]
data_table = data_table.drop(columns=["birthYear"])
data_table = data_table.to_json(orient="records", force_ascii=False)

prompt = f"""
Please extract the following information from the data:
1. First Name (name)
2. Last Name (surname)
3. Gender (gender)
4. Birth Year (born)
5. Birth City (city)
6. Job Profile Tags (tags) - based on the person's job profile, can be one
or multiple tags for person. But obligatory must be at least one tag for
each person. The tags should be selected from the following list:
'IT' - osoba pracuje w branży informatycznej, np. programista, administrator sieci,
'transport' - osoba związana z pracą w transporcie, np. kierowca, logistyk,
'edukacja' - osoba pracuje w branży edukacyjnej, np. nauczyciel, prowadzący szkolenia,
'medycyna' - osoba pracuje w branży medycznej, np. lekarz, fizjoterapeuta, sprzedawca leków,
'praca z ludźmi' - bezpośredni kontakt z klientem, np. sprzedawca, nauczyciel, lekarz, psycholog,
'praca z pojazdami' - praca związana z pojazdami, np. kierowca, mechanik samochodowy, kasjer biletów,
'praca z ludźmi' - praca związana z obsługą klientów w miejscach publicznych,
'praca fizyczna' - praca wymagająca wysiłku fizycznego, np. budowlaniec, rolnik.
Data to be extracted, containing multiple people: {data_table}
"""


class Person(BaseModel):
    name: str = Field(description="The first name of the person.")
    surname: str = Field(description="The last name of the person.")
    gender: str = Field(description="The gender of the person.")
    born: int = Field(description="The birth year of the person.")
    city: str = Field(description="The city where the person was born.")
    tags: List[str] = Field(
        description="""One or multiple tags related to the person's job profile"""
    )


class Answer(BaseModel):
    people: List[Person] = Field(
        description="A list of people extracted from the data."
    )


client = genai.Client()

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt,
    config={
        "response_mime_type": "application/json",
        "response_json_schema": Answer.model_json_schema(),
    },
)

people_json = Answer.model_validate_json(response.text)

transport_people = [
    person.model_dump() for person in people_json.people if "transport" in person.tags
]

output_dict = {
    "apikey": API_KEY,
    "task": "people",
    "answer": transport_people,
}

url = "https://hub.ag3nts.org/verify"

try:
    response = requests.post(url, json=output_dict)

    if response.status_code == 200:
        print("Sukces!")
        print(f"Odpowiedź serwera: {response.json()}")
    else:
        print(f"Błąd: Kod statusu {response.status_code}")
        print(f"Szczegóły: {response.text}")

except requests.exceptions.RequestException as e:
    print(f"Wystąpił problem z połączeniem: {e}")

import os
import io
import json
import requests
from PIL import Image
from google import genai

# Configuration
API_KEY = os.environ.get("AIDEVS_API_KEY")
BASE_URL = "https://hub.ag3nts.org"

# Initialize Google GenAI client
client = genai.Client()


def reset_and_fetch_board():
    """
    Resets the electricity puzzle board and fetches the current state of the board as
    an image.

    Returns:
    - A PIL Image object of the current board state.
    """

    url = f"{BASE_URL}/data/{API_KEY}/electricity.png?reset=1"
    res = requests.get(url)

    # Verify that the response is actually an image (Content-Type contains 'image')
    if "image" not in res.headers.get("Content-Type", ""):
        print(f"Error! The server did not return an image. Server response: {res.text}")
        raise Exception("Received text instead of an image. Check your API_KEY.")

    return Image.open(io.BytesIO(res.content))


def fetch_solved_reference():
    """
    Fetches the reference image of the solved electricity puzzle board.

    Returns:
    - A PIL Image object of the solved board state.
    """

    solved_url = f"{BASE_URL}/i/solved_electricity.png"
    res = requests.get(solved_url)

    if "image" not in res.headers.get("Content-Type", ""):
        print(f"Error! Failed to download the solution reference: {res.text}")
        return None

    return Image.open(io.BytesIO(res.content))


def analyze_board_with_vision(current_img, solved_img):
    """
    Analyzes the current and solved board images using Google Gemini Vision to determine
    the number of clockwise rotations needed for each tile to match the solved state.
    Returns a dictionary with the coordinates of each tile and the number of rotations needed.

    Parameters:
    - current_img: PIL Image object of the current board state.
    - solved_img: PIL Image object of the solved board state.

    Returns:
    - A dictionary with keys as tile coordinates (e.g., "1x1", "1x2", etc.) and values as the
    number of clockwise rotations needed (0, 1, 2, or 3).
    """

    prompt = """
    You are an expert in solving rotating tile puzzles (electrical paths).
    You have two images:
    1. 'electricity.png' - the current state of the 3x3 board.
    2. 'solved_electricity.png' - the target solved state.

    Rules:
    - The board is arranged row-column from 1x1 to 3x3.
    - Each tile can be rotated clockwise by 90 degrees (rotation values: 0 = no rotation, 1 = 90°, 2 = 180°, 3 = 270°).
    - Compare each tile from the current state with the corresponding tile in the target state.
    - Determine the exact number of clockwise rotations (0, 1, 2, or 3) needed for each tile so it matches the target state exactly.

    Return ONLY a plain JSON object (without any extra text or markdown like ```json):
    {
      "1x1": number,
      "1x2": number,
      "1x3": number,
      "2x1": number,
      "2x2": number,
      "2x3": number,
      "3x1": number,
      "3x2": number,
      "3x3": number
    }
    """

    # Use the gemini-3.1-pro-preview model instead of Flash for maximum visual precision
    response = client.models.generate_content(
        model="gemini-3.1-pro-preview", contents=[prompt, current_img, solved_img]
    )

    text = response.text.strip()
    # Remove markdown fences if the model adds them
    if text.startswith("```json"):
        text = text[7:-3].strip()
    elif text.startswith("```"):
        text = text[3:-3].strip()

    return json.loads(text)


def execute_rotations(rotations_dict):
    """
    Executes the rotations on the electricity puzzle board by sending requests to the hub.

    Parameters:
    - rotations_dict: A dictionary with keys as tile coordinates (e.g., "1x1", "1x2", etc.)
    and values as the number of clockwise rotations needed (0, 1, 2, or 3).

    Returns:
    - True if the flag was found in any of the responses, False otherwise.
    """

    verify_url = f"{BASE_URL}/verify"

    for coord, turns in rotations_dict.items():
        if turns > 0:
            print(f"Rotating {turns} time(s) for tile {coord}...")
            for i in range(turns):
                payload = {
                    "apikey": API_KEY,
                    "task": "electricity",
                    "answer": {"rotate": coord},
                }
                res = requests.post(verify_url, json=payload)
                data = res.json()

                # Print entire response for debugging purposes
                print(f"  -> Rotation result for {coord} (#{i + 1}): {data}")

                # Check whether any value in the response contains the flag marker
                if any("FLG" in str(v) for v in data.values()):
                    print("\nSUCCESS! FLAG FOUND:", data)
                    return True

    print("\nAll planned rotations have been sent.")
    return False


if __name__ == "__main__":

    print("1. Fetching the current board and reference...")
    current_board = reset_and_fetch_board()
    solved_board = fetch_solved_reference()

    print("2. Analyzing the board with Gemini Vision...")
    rotations = analyze_board_with_vision(current_board, solved_board)
    print("Computed rotations:", rotations)

    print("3. Sending moves to the hub...")
    success = execute_rotations(rotations)

    if not success:
        print(
            "\nCheck the final state or try again if the model misinterpreted the rotations."
        )

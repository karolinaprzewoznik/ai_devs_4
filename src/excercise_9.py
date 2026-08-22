import asyncio
import os
import re
import httpx

API_KEY = os.environ.get("AIDEVS_API_KEY")
BASE_URL = "https://hub.ag3nts.org"
ZMAIL_URL = f"{BASE_URL}/api/zmail"


class MailboxClient:
    """
    A client for interacting with the Zmail API to fetch emails and extract
    relevant data.
    """

    def __init__(self, http_client: httpx.AsyncClient, api_key: str) -> None:
        """
        Initialize the MailboxClient with an HTTP client and API key.

        Parameters:
            http_client (httpx.AsyncClient): An asynchronous HTTP client for
            making requests.
            api_key (str): The API key for authenticating with the Zmail API.
        """

        self.http_client = http_client
        self.api_key = api_key

    async def _call_zmail(self, action: str, **kwargs) -> dict:
        """
        Internal method to call the Zmail API with the specified action and
        parameters.

        Parameters:
            action (str): The action to perform on the Zmail API.
            **kwargs: Additional parameters to include in the API request.

        Returns:
            dict: The JSON response from the Zmail API, or an empty dictionary
            in case of an error.
        """

        payload = {"apikey": self.api_key, "action": action, **kwargs}

        try:
            await asyncio.sleep(0.8)
            response = await self.http_client.post(ZMAIL_URL, json=payload, timeout=15)
            return response.json()
        except Exception:
            return {}

    async def get_all_message_ids(self, max_pages: int = 10) -> set:
        """
        Fetch all message IDs from the Zmail inbox, up to a specified number of
        pages.

        Parameters:
            max_pages (int): The maximum number of pages to fetch from the inbox.

        Returns:
            set: A set of unique message IDs found in the inbox.
        """

        message_ids = set()

        for page in range(1, max_pages + 1):
            inbox = await self._call_zmail("getInbox", page=page, perPage=20)
            items = inbox.get("items", [])

            if not items:
                break

            for item in items:
                # Zmail can return message IDs under different keys, so we check
                # multiple possibilities
                msg_id = item.get("messageID") or item.get("id") or item.get("rowID")

                if msg_id:
                    message_ids.add(str(msg_id))

        return message_ids

    async def fetch_messages_batch(self, ids: list) -> list:
        """
        Fetch a batch of messages from the Zmail API based on a list of message
        IDs.

        Parameters:
            ids (list): A list of message IDs to fetch.

        Returns:
            list: A list of fetched messages.
        """

        result = await self._call_zmail("getMessages", ids=ids)

        return result.get("items", [])


class DataExtractor:
    """
    A utility class for extracting specific data from email content, such as
    passwords, confirmation codes, and dates.
    """

    @staticmethod
    def extract_password(content: str) -> str | None:
        """
        Extract a password from the email content if it contains specific
        keywords.

        Parameters:
            content (str): The email content to search for a password.

        Returns:
            str | None: The extracted password, or None if not found.
        """

        if any(kw in content.lower() for kw in ["hasło", "haslo", "password"]):

            match = re.search(
                r"(?:hasłem:|nowe:|to:)\s*([A-Za-z0-9_!@#$%^&*]+)", content
            )

            if match:
                return match.group(1).strip()

        return None

    @staticmethod
    def extract_confirmation_code(content: str) -> str | None:
        """
        Extract a confirmation code from the email content if it matches a
        specific pattern.

        Parameters:
            content (str): The email content to search for a confirmation code.

        Returns:
            str | None: The extracted confirmation code, or None if not found.
        """

        match = re.search(r"(SEC-[a-fA-F0-9]{32})", content)

        return match.group(1) if match else None

    @staticmethod
    def extract_date(content: str) -> str | None:
        """
        Extract a date from the email content if it matches a specific pattern.

        Parameters:
            content (str): The email content to search for a date.

        Returns:
            str | None: The extracted date in YYYY-MM-DD format, or None if not
            found.
        """

        match = re.search(r"\b(202\d-\d{2}-\d{2})\b", content)
        return match.group(1) if match else None


class CentralReporter:
    """
    A client for sending extracted data to a central reporting system.
    """

    def __init__(self, http_client: httpx.AsyncClient, api_key: str) -> None:
        """
        Initialize the CentralReporter with an HTTP client and API key.

        Parameters:
            http_client (httpx.AsyncClient): An asynchronous HTTP client for
            making requests.
            api_key (str): The API key for authenticating with the central
            reporting system.
        """

        self.http_client = http_client
        self.api_key = api_key

    async def send_answer(self, task: str, answer_data: dict) -> dict:
        """
        Send the extracted answer data to the central reporting system.

        Parameters:
            task (str): The task identifier for the data being sent.
            answer_data (dict): The extracted data to send.

        Returns:
            dict: The JSON response from the central reporting system.
        """

        payload = {"apikey": self.api_key, "task": task, "answer": answer_data}
        response = await self.http_client.post(
            f"{BASE_URL}/verify", json=payload, timeout=15
        )

        return response.json()


async def main():
    async with httpx.AsyncClient() as http_client:
        mail_client = MailboxClient(http_client, API_KEY)
        reporter = CentralReporter(http_client, API_KEY)

        print("Scanning the entire mailbox...")
        message_ids = list(await mail_client.get_all_message_ids())
        print(f"Found {len(message_ids)} message IDs.")

        password = None
        date = None
        confirmation_code = None

        # Process all messages in batches of 5
        for i in range(0, len(message_ids), 5):
            batch_ids = message_ids[i : i + 5]
            messages = await mail_client.fetch_messages_batch(batch_ids)

            for msg in messages:
                content = msg.get("message", "")

                if not password:
                    password = DataExtractor.extract_password(content)
                if not confirmation_code:
                    confirmation_code = DataExtractor.extract_confirmation_code(content)
                if not date:
                    date = DataExtractor.extract_date(content)

        print(
            f"Extraction result -> Password: {password}, Date: {date}, Code: {confirmation_code}"
        )

        if password and date and confirmation_code:
            print("We have all required data! Sending to the central system...")
            answer = {
                "password": password,
                "date": date,
                "confirmation_code": confirmation_code,
            }
            res = await reporter.send_answer("mailbox", answer)
            print("Central response:", res)
        else:
            print("Failed to complete all required fields.")


if __name__ == "__main__":
    asyncio.run(main())

import requests


class AIDOClient:
    def __init__(self, api_key, base_url="https://api.aido.com"):
        self.api_key = api_key
        self.base_url = base_url

    def _send_request(self, method, endpoint, json=None, params=None, stream=False):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        url = f"{self.base_url}/{endpoint}"
        response = requests.request(method, url, json=json, params=params, headers=headers, stream=stream)

        return response

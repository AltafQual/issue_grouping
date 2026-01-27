import importlib
import os

import requests

ISSUE_GROUPING_API_URL = os.getenv("ISSUE_GROUPING_API_URL", "http://hyd-lablnx904:8010")


def get_two_run_ids_cluster_info(run_id_a: str, run_id_b: str, timeout: int = 600) -> dict:
    """
    Calls the API to get regression for two run IDs.

    Args:
        run_id_a (str): Latest Run ID.
        run_id_b (str): Run ID to find regression against.

    Returns:
        dict: JSON response from the API or empty JSON in case of exception.
    """
    url = f"{ISSUE_GROUPING_API_URL}/api/get_two_run_ids_cluster_info/"
    headers = {"accept": "application/json", "Content-Type": "application/json"}
    payload = {"run_id_a": run_id_a, "run_id_b": run_id_b}

    try:
        print(f"POST: url: {url}, json: {payload}")
        response = requests.post(url, headers=headers, json=payload, timeout=timeout)
        response.raise_for_status()  # Raises HTTPError for bad responses
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error while fetch regression between 2 apis: {e}")
        return {}


async def get_two_run_ids_cluster_info_async(run_id_a: str, run_id_b: str, timeout: int = 600) -> dict:
    """
    Async version: Calls the API to get regression for two run IDs.

    Args:
        run_id_a (str): Latest Run ID.
        run_id_b (str): Run ID to find regression against.

    Returns:
        dict: JSON response from the API or empty JSON in case of exception.
    """

    # Check if aiohttp is installed
    if importlib.util.find_spec("aiohttp") is None:
        print("Warning: aiohttp module is not installed. Please install it using 'pip install aiohttp'.")
        return {}
    import aiohttp

    url = f"{ISSUE_GROUPING_API_URL}/api/get_two_run_ids_cluster_info/"
    headers = {"accept": "application/json", "Content-Type": "application/json"}
    payload = {"run_id_a": run_id_a, "run_id_b": run_id_b}

    try:
        print(f"POST: url: {url}, json: {payload}")
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload, timeout=timeout) as response:
                response.raise_for_status()
                return await response.json()
    except Exception as e:
        print(f"Error while fetching regression between 2 APIs: {e}")
        return {}

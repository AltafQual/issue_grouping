import json
import os
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urljoin

import aiohttp

from src.constants import GERRIT_API_CONFIG

_GERRIT_MAGIC_PREFIX = ")]}'\n"


class GerritClientAsync:
    def __init__(
        self,
        base_url: str = GERRIT_API_CONFIG.host,
        username: Optional[str] = GERRIT_API_CONFIG.user_name,
        password: Optional[str] = GERRIT_API_CONFIG.http_password,
        verify_ssl: bool = True,
        default_headers: Optional[Dict[str, str]] = None,
        default_params: Optional[Dict[str, Any]] = None,
    ):

        self.base_url = base_url.rstrip("/") + "/"
        self.auth = aiohttp.BasicAuth(username, password) if username and password else None
        self._api_prefix = "a/" if self.auth else ""
        self.verify_ssl = verify_ssl
        self._session: Optional[aiohttp.ClientSession] = None

        self.default_headers = {"Accept": "application/json"}
        if default_headers:
            self.default_headers.update(default_headers)
        self.default_params = default_params or {}

    async def __aenter__(self):
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def start(self) -> None:
        """Optionally create the session up front."""
        await self._ensure_session()

    async def close(self) -> None:
        """Close the session if it was created."""
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None

    async def _ensure_session(self) -> None:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(headers=self.default_headers)

    def _full_url(self, endpoint: str) -> str:
        endpoint = endpoint.lstrip("/")
        return urljoin(self.base_url, f"{self._api_prefix}{endpoint}")

    @staticmethod
    def _decode(text: str) -> Any:
        # Strip Gerrit XSSI prefix then decode JSON
        if text.startswith(_GERRIT_MAGIC_PREFIX):
            text = text[len(_GERRIT_MAGIC_PREFIX) :]
        return json.loads(text)

    async def _get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        await self._ensure_session()
        assert self._session is not None

        url = self._full_url(endpoint)
        # Merge default params with provided params
        q: Dict[str, Any] = dict(self.default_params)
        if params:
            # Gerrit allows multiple 'o' query params as a list
            # aiohttp will serialize list values into repeated params
            q.update(params)
        print(f"GET: url: {url}, params: {params}")
        async with self._session.get(url, params=q, auth=self.auth, ssl=self.verify_ssl) as resp:
            resp.raise_for_status()
            text = await resp.text()
            return self._decode(text)

    async def get_change(self, change_id: Union[int, str]) -> Dict[str, Any]:
        """
        GET /changes/{id}
        'change_id' can be numeric (_number) or the composite 'project~branch~Change-Id'.
        """
        return await self._get(f"/changes/{change_id}")

    async def get_change_detail(
        self, change_id: Union[int, str], options: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        GET /changes/{id}/detail
        Expand with 'o' options (e.g., CURRENT_REVISION, MESSAGES, LABELS, DETAILED_LABELS, CURRENT_COMMIT, ALL_FILES).
        If no options passed, uses a sensible default set.
        """
        if options is None:
            options = [
                "CURRENT_REVISION",  # include current revision info
                "CURRENT_COMMIT",  # include commit details for current revision
                "LABELS",  # include labels
                "DETAILED_LABELS",  # include detailed label info
                "MESSAGES",  # include change messages
                "DETAILED_ACCOUNTS",  # include richer account info
                "ALL_FILES",  # include all file summaries
            ]
        params = {"o": options} if options else None
        return await self._get(f"/changes/{change_id}/detail", params=params)

    async def list_comments(self, change_id: Union[int, str]) -> Dict[str, Any]:
        """
        GET /changes/{id}/comments
        Published comments only.
        """
        return await self._get(f"/changes/{change_id}/comments")

    async def list_revisions(self, change_id: Union[int, str]) -> Dict[str, Any]:
        """
        GET /changes/{id}/revisions/
        Map of revisions keyed by SHA.
        """
        return await self._get(f"/changes/{change_id}/revisions/")

    async def get_revision_commit(
        self, change_id: Union[int, str], revision: str, include_links: bool = False
    ) -> Dict[str, Any]:
        """
        GET /changes/{id}/revisions/{rev}/commit[?links]
        Use 'include_links=True' to include external web links.
        """
        params = {"links": ""} if include_links else None
        return await self._get(f"/changes/{change_id}/revisions/{revision}/commit", params=params)

    async def list_revision_files(self, change_id: Union[int, str], revision: str) -> Dict[str, Any]:
        """
        GET /changes/{id}/revisions/{rev}/files
        """
        return await self._get(f"/changes/{change_id}/revisions/{revision}/files")

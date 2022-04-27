import os
import html
import json
import requests
from bs4 import BeautifulSoup
from dataclasses import dataclass
from requests.auth import HTTPBasicAuth
from typing import Final, Optional

class ConfluenceRequester:
    """
    Retrieves information from Confluence pages given Confluence API credentials and a PageID.
    """

    CONTENT_API_URL_FORMAT: Final[str] = "https://confluence.nvidia.com/rest/api/content/{pageid}?expand=body.storage"
    DIRECT_PAGE_URL_FORMAT: Final[str] = "https://confluence.nvidia.com/display/{org_id}/{page_title}"

    def __init__(self, username: str, api_key: str):
        self.creds = HTTPBasicAuth(username, api_key)

    def get_page_contents(self, pageid: int) -> str:
        """
        Retrieves the contents of the specified PageID as a string.

        Args:
            pageid (int): The Confluence `pageid`.

                          This can be obtained by going to a Confluence page (i.e. https://confluence.nvidia.com/display/GCA/MLPerf+Inference+Tutorials),
                          clicking the 3 dots in the top right corner, next to 'Send Page' and below your profile icon,
                          and selecting 'Page Information'. This will bring you to a page like
                          "https://confluence.nvidia.com/pages/viewinfo.action?pageId=341529806". The pageID is
                          indicated in the URL as the GET parameter 'pageID'.

        Returns:
            str: The contents of the Confluence page body as an HTML string. If you are familiar with the Confluence
                 API, this corresponds to the keyspace "body.storage.value" in the returned JSON.

        Raises:
            requests.exceptions.ConnectionError:
                If the request itself has a network error.
            requests.exceptions.HTTPError:
                If the request completed successfully, but returned a non-200 HTTP status.
        """
        # Implicitly raises a requests.ConnectionError if a network connection error occurs
        resp = requests.get(ConfluenceRequester.CONTENT_API_URL_FORMAT.format(pageid=pageid), auth=self.creds)
        resp.raise_for_status()  # Raises an HTTPError if status is not HTTP_OK.
        data = json.loads(resp.content)
        return data["body"]["storage"]["value"]

    def get_page_id(self, page_title: str, org_id: str = "GCA") -> int:
        """
        Retrieves the Confluence page id given the title of the page and organization the page is under.

        Args:
            page_title (str): The title of the Confluence page as a raw string. Do *not* HTML-escape the string before
                              passing it in.
            org_id (str): The ID of the organization as an abbreviation. Default: "GCA" (GPU Compute Arch).

        Returns:
            int: The page id of the requested Confluence page.

        Raises:
            requests.exceptions.ConnectionError:
                If the request itself has a network error.
            requests.exceptions.HTTPError:
                If the request completed successfully, but returned a non-200 HTTP status.
        """
        url = ConfluenceRequester.DIRECT_PAGE_URL_FORMAT.format(org_id=org_id, page_title=html.escape(page_title))
        # Implicitly raises a requests.ConnectionError if a network connection error occurs
        resp = requests.get(url, auth=self.creds)
        # Parse response content to get ajs-page-id
        soup = BeautifulSoup(resp.content, "html.parser")
        page_id_candidates = soup.find_all("meta", {"name": "ajs-page-id"})
        if len(page_id_candidates) == 0:
            raise RuntimeError(f"No page ID found for {org_id}/{page_title}'")
        elif len(page_id_candidates) > 1:
            raise RuntimeError(f"Multiple page IDs found for {org_id}/{page_title}'")
        return int(page_id_candidates[0].get("content"))


@dataclass(frozen=True)
class ConfluenceInternalTagArgs:
    public_page: Optional[str] = None
    """str: Link to the public page to substitute in for the internal section. None if the section should not be
    replaced."""

    page_available: bool = False
    """bool: Whether or not 'public_page' is available to be used as a substitution. If True, the section will be
    replaced with the contents of 'public_page'. If False, the section will be replaced with a notification that the
    section is 'under construction'."""

    public_string: Optional[str] = None
    """str: A string use as a replacement for the section header or current line, kind of like the 'alt' HTML tag
    field."""


G_CONFLUENCE_USERNAME: Final[str] = os.environ.get("CONFLUENCE_API_USER")
G_CONFLUENCE_PASSWORD: Final[str] = os.environ.get("CONFLUENCE_API_PASS")
G_CONFLUENCE_REQUESTER = ConfluenceRequester(G_CONFLUENCE_USERNAME, G_CONFLUENCE_PASSWORD)


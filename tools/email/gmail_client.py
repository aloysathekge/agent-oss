"""Gmail OAuth and API resource factory.

Single responsibility: authenticate with Gmail and hand back an authenticated
googleapiclient `Resource`. All email tools import `get_service` from here so
that auth logic lives in exactly one place.

On first call, opens a browser for OAuth consent and writes `token.json`.
Subsequent calls reuse the cached token (refreshing if expired).
"""
import os

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


# Read + send + compose. Compose is required for the drafts API.
SCOPES = [
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.compose",
]

# OAuth client secret from Google Cloud Console (Desktop app type).
CREDS_PATH = os.environ.get("GMAIL_CREDENTIALS_PATH", "credentials.json")

# Cached user consent token. Auto-created after first successful OAuth flow.
TOKEN_PATH = "token.json"


def get_service():
    """Return an authenticated Gmail API resource.

    First call triggers browser-based OAuth consent. Later calls reuse the
    saved token. If the token is expired but has a refresh token, it is
    refreshed silently.
    """
    creds = None

    if os.path.exists(TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(CREDS_PATH):
                raise FileNotFoundError(
                    f"Gmail OAuth client secret not found at '{CREDS_PATH}'. "
                    "Download it from Google Cloud Console (Desktop app OAuth client) "
                    "and set GMAIL_CREDENTIALS_PATH in your .env, or place it at ./credentials.json."
                )
            flow = InstalledAppFlow.from_client_secrets_file(CREDS_PATH, SCOPES)
            creds = flow.run_local_server(port=0)

        with open(TOKEN_PATH, "w") as f:
            f.write(creds.to_json())

    return build("gmail", "v1", credentials=creds)

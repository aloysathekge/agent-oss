"""Google Calendar OAuth and API resource factory.

Mirrors tools/email/gmail_client.py but for Calendar v3. Uses a separate
token file (token_calendar.json) so the two skills stay fully independent —
enabling or disabling either skill doesn't affect the other's auth state.
"""
import os

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


SCOPES = ["https://www.googleapis.com/auth/calendar"]

CREDS_PATH = os.environ.get("GMAIL_CREDENTIALS_PATH", "credentials.json")
TOKEN_PATH = "token_calendar.json"


def get_service():
    """Return an authenticated Google Calendar API resource.

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
                    f"OAuth client secret not found at '{CREDS_PATH}'. "
                    "Download it from Google Cloud Console (Desktop app OAuth client) "
                    "and set GMAIL_CREDENTIALS_PATH in your .env, or place it at ./credentials.json."
                )
            flow = InstalledAppFlow.from_client_secrets_file(CREDS_PATH, SCOPES)
            creds = flow.run_local_server(port=0)

        with open(TOKEN_PATH, "w") as f:
            f.write(creds.to_json())

    return build("calendar", "v3", credentials=creds)

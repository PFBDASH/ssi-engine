# middleware.py
import os
import requests
from flask import request, redirect, make_response, g

MEMBERSTACK_API_KEY = os.getenv("MEMBERSTACK_API_KEY", "").strip()
WEBFLOW_LOGIN_URL = os.getenv("WEBFLOW_LOGIN_URL", "https://ssi-auth.webflow.io/sign-in").strip()

VERIFY_URL = "https://admin.memberstack.com/members/verify-token"


def verify_memberstack_token(token: str) -> dict | None:
    """
    Returns the verified member payload dict if valid, otherwise None.
    """
    if not token or not MEMBERSTACK_API_KEY:
        return None

    try:
        r = requests.post(
            VERIFY_URL,
            headers={
                "X-API-KEY": MEMBERSTACK_API_KEY,
                "Content-Type": "application/json",
            },
            json={"token": token},
            timeout=10,
        )
        if r.status_code != 200:
            return None

        data = r.json()
        # Memberstack returns a "member" object on success (common pattern).
        # If your response shape differs, keep this flexible.
        return data.get("member") or data
    except Exception:
        return None


def attach_auth(app, protected_prefixes=("/api", "/app")):
    """
    Call this once from app.py after app = Flask(__name__)
    Protects /app and /api* by verifying Memberstack token.
    Accepts token from:
      1) querystring ?ms=...
      2) cookie "ms_token"
    """
    @app.before_request
    def _auth_gate():
        path = request.path or "/"
        if not any(path.startswith(p) for p in protected_prefixes):
            return None

        # 1) Grab token from querystring on first hit
        token = request.args.get("ms", "").strip()

        # 2) Otherwise use cookie for all subsequent requests
        if not token:
            token = request.cookies.get("ms_token", "").strip()

        member = verify_memberstack_token(token)
        if not member:
            # Clear cookie and bounce to Webflow login
            resp = make_response(redirect(WEBFLOW_LOGIN_URL))
            resp.set_cookie("ms_token", "", expires=0, path="/")
            return resp

        # Stash for your routes to use if needed
        g.member = member
        g.ms_token = token
        return None
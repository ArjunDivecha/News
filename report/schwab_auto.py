#!/usr/bin/env python3
"""
=============================================================================
SCRIPT NAME: schwab_auto.py
=============================================================================

INPUT FILES:
    (reads from env vars set by .env via config.py):
    - SCHWAB_USERNAME      : Schwab login ID
    - SCHWAB_PASSWORD      : Schwab account password
    - SCHWAB_TOTP_SECRET   : TOTP shared secret (Base32) for 2FA code generation
    - SCHWAB_APP_KEY       : Schwab developer app client ID
    - SCHWAB_APP_SECRET    : Schwab developer app secret

OUTPUT FILES:
    (none - returns the OAuth callback URL string)
    Side effect: schwabdev writes refreshed tokens to ~/.schwabdev/tokens.db

VERSION: 1.0
LAST UPDATED: 2026-06-11
AUTHOR: Arjun Divecha

DESCRIPTION:
    Automated Schwab OAuth token refresh using Playwright headless Chromium.
    Designed to be passed as schwabdev's call_on_auth callback:

        from schwab_auto import get_auth_function
        client = schwabdev.Client(app_key, app_secret,
                                  call_on_auth=get_auth_function())

    When schwabdev needs a fresh refresh token (every ~7 days), it calls
    this function with the OAuth authorization URL. The function then:

      1. Launches headless Chromium
      2. Navigates to the Schwab OAuth authorization page
      3. Fills in username → password → TOTP code (if prompted)
      4. Waits for the redirect to the callback URL containing ?code=...
      5. Returns the full redirect URL so schwabdev can extract the code

    If SCHWAB_USERNAME / SCHWAB_PASSWORD / SCHWAB_TOTP_SECRET are not
    set, a RuntimeError is raised so holdings.py can fall back to the
    interactive flow.

DEPENDENCIES:
    - playwright (Chromium browser)
    - pyotp

USAGE:
    from schwab_auto import get_auth_function
    fn = get_auth_function()  # reads creds from env
    client = schwabdev.Client(app_key, app_secret, call_on_auth=fn)
=============================================================================
"""

import os
import sys
import time
from pathlib import Path
from typing import Callable

# ---------------------------------------------------------------------------
# Selectors for Schwab's login pages (stable as of June 2026)
# ---------------------------------------------------------------------------
# Schwab's OAuth flow uses a multi-page wizard:
#   Page 1: Login ID (username)
#   Page 2: Password
#   Page 3: TOTP verification code (if 2FA enabled)

# Login ID page
SEL_LOGIN_ID = 'input[name="loginId"], input#loginIdInput, input[type="text"]'

# Password page
SEL_PASSWORD = 'input[name="password"], input#passwordInput, input[type="password"]'

# TOTP page (appears after successful password if 2FA is on)
SEL_TOTP = 'input[name="totp"], input#totpInput, input[aria-label*="code"], input[inputmode="numeric"]'

# "Remember this device" checkbox (skip 2FA next time on this browser)
SEL_REMEMBER = 'input[name="remember"], input#rememberDevice, input[type="checkbox"]'

# Submit / Continue button
SEL_SUBMIT = (
    'button[type="submit"], '
    'button:has-text("Continue"), button:has-text("Log in"), '
    'button:has-text("Sign in"), '
    'input[type="submit"]'
)

# Common wait-after-click delay (seconds)
_CLICK_WAIT_S = 2.0

# Maximum time for the entire OAuth flow (seconds)
_FLOW_TIMEOUT_S = 90


def _find_and_fill(page, selectors: str, value: str, timeout_s: float = 8.0):
    """
    Find the first matching element from a comma-separated list of CSS
    selectors, click it, fill it, and press Enter.
    """
    sel_list = [s.strip() for s in selectors.split(",") if s.strip()]
    for sel in sel_list:
        try:
            el = page.wait_for_selector(sel, timeout=timeout_s * 1000, state="visible")
            if el:
                el.click()
                el.fill("")
                el.type(value, delay=50)  # human-like typing
                page.keyboard.press("Enter")
                return True
        except Exception:
            continue
    return False


def _click_submit(page, timeout_s: float = 5.0):
    """
    Click the first visible submit/continue button.
    """
    sel_list = [s.strip() for s in SEL_SUBMIT.split(",") if s.strip()]
    for sel in sel_list:
        try:
            btn = page.wait_for_selector(sel, timeout=timeout_s * 1000, state="visible")
            if btn:
                btn.click()
                time.sleep(_CLICK_WAIT_S)
                return True
        except Exception:
            continue
    # Last resort: press Enter
    page.keyboard.press("Enter")
    time.sleep(_CLICK_WAIT_S)
    return False


def _try_check_remember(page, timeout_s: float = 3.0):
    """Check 'remember this device' if present (reduces future 2FA)."""
    sel_list = [s.strip() for s in SEL_REMEMBER.split(",") if s.strip()]
    for sel in sel_list:
        try:
            cb = page.wait_for_selector(sel, timeout=timeout_s * 1000, state="visible")
            if cb and not cb.is_checked():
                cb.check()
                return True
        except Exception:
            continue
    return False


def schwab_oauth(auth_url: str, username: str, password: str,
                 totp_secret: str, callback_url: str) -> str:
    """
    Run the full Schwab OAuth flow in headless Chromium.

    Args:
        auth_url: The full OAuth authorization URL from schwabdev.
        username: Schwab login ID.
        password: Schwab account password.
        totp_secret: Base32 TOTP shared secret for 2FA code generation.
        callback_url: The redirect_uri (e.g. "https://127.0.0.1").

    Returns:
        The full redirect URL containing ?code=... and &session=...
    """
    import pyotp
    from playwright.sync_api import sync_playwright, TimeoutError as PwTimeout

    totp = pyotp.TOTP(totp_secret)

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"],
        )
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            ),
        )
        page = context.new_page()

        try:
            # --- Step 1: Navigate to Schwab OAuth authorization ---
            page.goto(auth_url, wait_until="domcontentloaded", timeout=30000)
            time.sleep(2)  # let JS redirects settle

            # --- Step 2: Login ID (username) ---
            if not _find_and_fill(page, SEL_LOGIN_ID, username):
                # Some flows present username and password on the same page
                pass
            time.sleep(1)

            # --- Step 3: Password ---
            # Schwab sometimes auto-advances, sometimes needs a click
            page.wait_for_load_state("networkidle", timeout=10000)
            _click_submit(page)  # click Continue after username if needed
            time.sleep(2)

            if not _find_and_fill(page, SEL_PASSWORD, password):
                # Maybe it was same-page; try again
                _find_and_fill(page, SEL_PASSWORD, password)
            time.sleep(1)

            # --- Step 4: Submit password ---
            page.wait_for_load_state("networkidle", timeout=10000)
            _click_submit(page)
            time.sleep(3)

            # --- Step 5: TOTP (if prompted) ---
            page.wait_for_load_state("networkidle", timeout=10000)
            _try_check_remember(page)

            if _find_and_fill(page, SEL_TOTP, totp.now(), timeout_s=5.0):
                time.sleep(1)
                _try_check_remember(page)
                _click_submit(page)
                time.sleep(3)

            # --- Step 6: Wait for redirect to callback URL ---
            # The callback URL is https://127.0.0.1 (or similar), which will
            # fail to load — Playwright catches this as the final URL.
            try:
                page.wait_for_url(
                    f"{callback_url}**",
                    timeout=_FLOW_TIMEOUT_S * 1000,
                )
            except PwTimeout:
                # If the page didn't redirect, check if code is in current URL
                current = page.url
                if "code=" in current:
                    pass  # already at the right URL
                else:
                    raise RuntimeError(
                        f"OAuth flow timed out waiting for redirect to "
                        f"{callback_url}. Current URL: {current}"
                    )

            redirect_url = page.url
            if "code=" not in redirect_url:
                raise RuntimeError(
                    f"Redirect URL missing authorization code: {redirect_url}"
                )

            print(f"    Schwab: OAuth flow complete, got authorization code")
            return redirect_url

        except Exception as e:
            # Take a screenshot for debugging before closing
            try:
                debug_dir = Path(__file__).resolve().parent.parent / "outputs" / "debug"
                debug_dir.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=str(debug_dir / "schwab_oauth_failure.png"))
                print(f"    Schwab OAuth screenshot saved: "
                      f"{debug_dir / 'schwab_oauth_failure.png'}")
            except Exception:
                pass
            raise RuntimeError(f"Schwab OAuth automation failed: {e}")

        finally:
            browser.close()


def get_auth_function(callback_url: str = "https://127.0.0.1") -> Callable:
    """
    Return a callable suitable for schwabdev.Client(call_on_auth=...).

    Reads SCHWAB_USERNAME, SCHWAB_PASSWORD, SCHWAB_TOTP_SECRET from
    environment (set in .env). Raises RuntimeError if any are missing
    so the caller can fall back to the interactive flow.

    The returned function has signature fn(auth_url: str) -> str.
    """
    username = os.getenv("SCHWAB_USERNAME")
    password = os.getenv("SCHWAB_PASSWORD")
    totp_secret = os.getenv("SCHWAB_TOTP_SECRET")

    missing = []
    if not username:
        missing.append("SCHWAB_USERNAME")
    if not password:
        missing.append("SCHWAB_PASSWORD")
    if not totp_secret:
        missing.append("SCHWAB_TOTP_SECRET")

    if missing:
        raise RuntimeError(
            f"Schwab auto-auth not configured: missing env vars: "
            f"{', '.join(missing)}")

    def _auth(auth_url: str) -> str:
        return schwab_oauth(
            auth_url, username, password, totp_secret, callback_url)

    return _auth

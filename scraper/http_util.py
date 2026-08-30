"""Shared HTTP helpers with retry/backoff for flaky research APIs."""

import logging
import time

import requests

logger = logging.getLogger(__name__)


def get_json(session, url, params=None, timeout=45, max_retries=6, retry_statuses=(429, 500, 502, 503, 504)):
    """
    GET url and return parsed JSON. Retries on timeouts, connection errors,
    and retry_statuses (notably 429). Raises the last HTTPError / RequestException
    if all attempts fail.
    """
    delay = 1.0
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = session.get(url, params=params, timeout=timeout)
        except requests.RequestException as exc:
            last_exc = exc
            logger.warning("Request failed (%s) attempt %d/%d — retrying in %.1fs", exc, attempt, max_retries, delay)
            time.sleep(delay)
            delay = min(delay * 2, 60)
            continue

        if resp.status_code in retry_statuses:
            retry_after = resp.headers.get("Retry-After")
            try:
                wait = float(retry_after) if retry_after is not None else delay * 2
            except (TypeError, ValueError):
                wait = delay * 2
            wait = max(wait, delay)
            logger.warning("HTTP %d from %s — backing off %.1fs (attempt %d/%d)", resp.status_code, url, wait, attempt, max_retries)
            time.sleep(wait)
            delay = min(delay * 2, 60)
            continue

        try:
            resp.raise_for_status()
        except requests.HTTPError as exc:
            body = (resp.text or "")[:500]
            logger.error("HTTP %d from %s: %s", resp.status_code, url, body)
            raise
        return resp.json()

    if last_exc:
        raise last_exc
    raise RuntimeError(f"GET {url} failed after {max_retries} retries")


def post_json(session, url, payload, params=None, timeout=45, max_retries=6, retry_statuses=(429, 500, 502, 503, 504)):
    delay = 1.0
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = session.post(url, json=payload, params=params, timeout=timeout)
        except requests.RequestException as exc:
            last_exc = exc
            logger.warning("POST failed (%s) attempt %d/%d — retrying in %.1fs", exc, attempt, max_retries, delay)
            time.sleep(delay)
            delay = min(delay * 2, 60)
            continue

        if resp.status_code in retry_statuses:
            retry_after = resp.headers.get("Retry-After")
            try:
                wait = float(retry_after) if retry_after is not None else delay * 2
            except (TypeError, ValueError):
                wait = delay * 2
            wait = max(wait, delay)
            logger.warning("HTTP %d POST %s — backing off %.1fs (attempt %d/%d)", resp.status_code, url, wait, attempt, max_retries)
            time.sleep(wait)
            delay = min(delay * 2, 60)
            continue

        resp.raise_for_status()
        return resp.json()

    if last_exc:
        raise last_exc
    raise RuntimeError(f"POST {url} failed after {max_retries} retries")

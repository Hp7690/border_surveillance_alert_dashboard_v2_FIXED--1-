"""
webhooks.py
------------
Lightweight webhook / C2 (command-and-control) integration: register
one or more external URLs that should receive an HTTP POST with the
JSON payload of every alert, in real time. This is what lets an
external system -- a real C2 platform, a SIEM, a Slack/Teams bridge, a
ticketing system -- subscribe to this platform's alerts WITHOUT having
to poll the REST API.

Webhooks are persisted in webhooks.json (created automatically) so
they survive restarts, and can be managed at runtime via:
    GET    /api/webhooks              list registered webhooks
    POST   /api/webhooks {"url":...}  register a new one
    DELETE /api/webhooks/<id>         remove one

Delivery is BEST-EFFORT and fire-and-forget: a slow or dead endpoint
must never slow down or crash the live surveillance pipeline. Failed
deliveries are logged, not retried -- for guaranteed delivery in a real
deployment, put a durable queue (e.g. Redis/RabbitMQ) in front of this.
"""

import json
import os
import uuid

import eventlet
import requests

CONFIG_PATH = "webhooks.json"
POST_TIMEOUT_SECONDS = 4


class WebhookManager:
    def __init__(self, config_path=CONFIG_PATH):
        self.config_path = config_path
        self._webhooks = self._load()  # list of {"id": ..., "url": ...}

    def _load(self):
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path) as f:
                    return json.load(f)
            except Exception as e:
                print(f"[WebhookManager] Failed to load {self.config_path}: {e}")
        return []

    def _save(self):
        with open(self.config_path, "w") as f:
            json.dump(self._webhooks, f, indent=2)

    def list(self):
        return list(self._webhooks)

    def add(self, url):
        entry = {"id": str(uuid.uuid4()), "url": url}
        self._webhooks.append(entry)
        self._save()
        print(f"[WebhookManager] Registered webhook: {url}")
        return entry

    def remove(self, webhook_id):
        before = len(self._webhooks)
        self._webhooks = [w for w in self._webhooks if w["id"] != webhook_id]
        if len(self._webhooks) < before:
            self._save()
            return True
        return False

    def notify(self, alert_dict):
        """Fire-and-forget POST of the alert JSON to every registered webhook."""
        for w in self._webhooks:
            eventlet.spawn(self._post, w["url"], alert_dict)

    @staticmethod
    def _post(url, payload):
        try:
            requests.post(url, json=payload, timeout=POST_TIMEOUT_SECONDS)
        except Exception as e:
            print(f"[WebhookManager] Delivery to {url} failed: {e}")

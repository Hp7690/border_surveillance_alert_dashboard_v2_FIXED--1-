"""
auth.py
--------
Minimal username/password authentication for the dashboard --
border-security software should not be left open on the network with
no login.

Users are stored in data/users.json as {username: password_hash}, with
passwords hashed via werkzeug's PBKDF2 implementation (already a Flask
dependency -- no extra package to install). There is NO built-in
default account: create the first one with:

    python create_user.py --username admin

(You'll be prompted for a password interactively; it's hashed before
being written to disk and is never stored or logged in plaintext.)

SCOPE NOTE: this is intentionally simple -- a single shared login
mechanism, no roles/permissions, no rate-limiting/lockout, no 2FA.
It's adequate to keep the dashboard off the open network for a demo or
pilot deployment. A real production rollout should sit this behind a
proper identity provider (SSO/LDAP/etc.) and add rate-limiting.
"""
import json
import os

from werkzeug.security import check_password_hash, generate_password_hash

USERS_PATH = os.path.join("data", "users.json")


def _load():
    if os.path.exists(USERS_PATH):
        with open(USERS_PATH) as f:
            return json.load(f)
    return {}


def _save(users):
    os.makedirs(os.path.dirname(USERS_PATH), exist_ok=True)
    with open(USERS_PATH, "w") as f:
        json.dump(users, f, indent=2)


def has_any_users():
    return len(_load()) > 0


def create_user(username, password):
    users = _load()
    users[username] = generate_password_hash(password)
    _save(users)


def verify_user(username, password):
    if not username or not password:
        return False
    users = _load()
    hashed = users.get(username)
    if not hashed:
        return False
    return check_password_hash(hashed, password)


def delete_user(username):
    users = _load()
    if username in users:
        del users[username]
        _save(users)
        return True
    return False


def list_usernames():
    return list(_load().keys())

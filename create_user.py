"""
create_user.py
----------------
Create (or reset the password for) a dashboard login account.

Usage:
    python create_user.py --username admin

You'll be prompted for a password (hidden -- not echoed to the
terminal, and never stored or logged in plaintext). Run this once
before starting app.py for the first time: with zero accounts, nobody
can log into the dashboard.

To remove an account:
    python create_user.py --username admin --delete
"""
import argparse
import getpass

from core.auth import create_user, delete_user, list_usernames

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", required=True)
    parser.add_argument("--delete", action="store_true", help="Remove this account instead of creating it")
    parser.add_argument("--visible", action="store_true",
                         help="Show the password as you type it, instead of hiding it. "
                              "Use this if hidden input isn't working in your terminal "
                              "(e.g. some IDE-integrated terminals).")
    args = parser.parse_args()

    if args.delete:
        ok = delete_user(args.username)
        print(f"Removed '{args.username}'." if ok else f"No such user: '{args.username}'.")
        print(f"Remaining accounts: {list_usernames()}")
        raise SystemExit(0)

    if args.visible:
        password = input("Password (visible): ")
        confirm = input("Confirm password (visible): ")
    else:
        password = getpass.getpass("Password: ")
        confirm = getpass.getpass("Confirm password: ")
    if password != confirm:
        print("Passwords do not match -- aborted.")
        raise SystemExit(1)
    if len(password) < 6:
        print("Password too short -- use at least 6 characters. Aborted.")
        raise SystemExit(1)

    create_user(args.username, password)
    print(f"User '{args.username}' created/updated.")
    print(f"Existing accounts: {list_usernames()}")

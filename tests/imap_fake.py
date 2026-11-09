"""A tiny in-memory IMAP server for poll tests: SEARCH (everything), batched UID FETCH of headers, single body fetch.
Every SELECT must be read-only — the app never changes your mailbox flags."""

from __future__ import annotations

import re


def fake_imap(mail: dict[str, dict[bytes, bytes]]):
    class FakeIMAP:
        fetched: list[bytes] = []

        def __init__(self, *a, **k):
            self.folder = None

        def login(self, u, p):
            return "OK", []

        def list(self):
            return "OK", [f'(\\HasNoChildren) "/" "{name}"'.encode() for name in mail] + [b'(\\Trash) "/" "Papierkorb"']

        def select(self, name, readonly=False):
            assert readonly, "the mailbox must be opened read-only"
            self.folder = name.strip('"')
            return "OK", [str(len(mail[self.folder])).encode()]

        def uid(self, cmd, *args):
            box = mail[self.folder]
            if cmd == "search":
                return "OK", [b" ".join(box)]
            ids = args[0].split(b",")
            if "HEADER.FIELDS" in args[1]:
                out = []
                for u in ids:
                    head = re.split(rb"\r?\n\r?\n", box[u], maxsplit=1)[0]
                    out += [(b"%s (UID %s BODY[HEADER.FIELDS] {%d}" % (u, u, len(head)), head + b"\r\n\r\n"), b")"]
                return "OK", out
            FakeIMAP.fetched.append(args[0])
            return "OK", [(b"%s (UID %s BODY[] {1}" % (args[0], args[0]), box[args[0]]), b")"]

        def logout(self):
            return "OK", []

    return FakeIMAP

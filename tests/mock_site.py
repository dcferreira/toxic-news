# SPDX-FileCopyrightText: 2023-present Daniel Ferreira <daniel.ferreira.1@gmail.com>
#
# SPDX-License-Identifier: MIT

"""A stand-in for the newspaper front pages, so the pipeline runs offline.

The pipeline fetches every outlet from a single origin when it is given a base
URL — `https://bbc.com` is fetched from `<base_url>/bbc.com` — so this server
only has to map one outlet slug back to its recorded page, `<root>/<slug>.html`.
Nothing else is served, and no request ever reaches a real newspaper.

It is deliberately stdlib-only and imports nothing from the project or its
siblings, so this single file can be mounted (or copied into a container) on its
own. Run it as `python tests/mock_site.py` or `python -m tests.mock_site`, or
call `serve` to get the same server on a background thread.
"""

import argparse
import logging
import sys
import threading
from collections.abc import Callable, Mapping
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import cast
from urllib.parse import urlsplit

logger = logging.getLogger("mock_site")

#: The recorded front pages: `tests/assets/html`, next to this file.
DEFAULT_ROOT = Path(__file__).resolve().parent / "assets" / "html"

#: The only HTTP methods the mock answers; every other one is refused.
ALLOWED_METHODS = ("GET", "HEAD")

HTML_CONTENT_TYPE = "text/html; charset=utf-8"
TEXT_CONTENT_TYPE = "text/plain; charset=utf-8"


class FixtureServer(ThreadingHTTPServer):
    """A server that answers from the recorded front pages under `root`."""

    def __init__(self, server_address: tuple[str, int], root: Path) -> None:
        """Bind to `server_address`, serving the fixtures under `root`.

        `root` is resolved here, so that serving does not depend on the working
        directory of whoever asked: requests are handled on another thread, by
        which time a relative path would mean something else.
        """
        self.root = root.resolve()
        super().__init__(server_address, FixtureHandler)


class FixtureHandler(BaseHTTPRequestHandler):
    """Serve one recorded front page per outlet slug, and nothing else."""

    def do_GET(self) -> None:
        """Serve the front page the requested slug names."""
        slug = urlsplit(self.path).path.strip("/")
        path = self._fixtures_root() / f"{slug}.html"
        if not _is_fixture_name(slug) or not path.is_file():
            self._not_found(slug)
            return
        self._respond(HTTPStatus.OK, HTML_CONTENT_TYPE, path.read_bytes())

    def do_HEAD(self) -> None:
        """Serve what GET would, without the body."""
        self.do_GET()

    def __getattr__(self, name: str) -> Callable[[], None]:
        """Refuse every method other than GET and HEAD.

        `BaseHTTPRequestHandler` dispatches on `do_<METHOD>`, and answers a
        method it has no `do_` for with a 501. Routing them all here makes the
        refusal a 405 Method Not Allowed instead, which is what a client
        sending, say, POST to a static mock should be told.
        """
        if name.startswith("do_"):
            return self._method_not_allowed
        raise AttributeError(name)

    def _fixtures_root(self) -> Path:
        """Return the directory of fixtures the server was started for."""
        return cast("FixtureServer", self.server).root

    def _not_found(self, slug: str) -> None:
        """Answer with 404, naming the slug that has no recorded page."""
        self._respond(
            HTTPStatus.NOT_FOUND,
            TEXT_CONTENT_TYPE,
            f"No recorded front page for {slug!r}\n".encode(),
        )

    def _method_not_allowed(self) -> None:
        """Answer with 405 Method Not Allowed."""
        allowed = " or ".join(ALLOWED_METHODS)
        self._respond(
            HTTPStatus.METHOD_NOT_ALLOWED,
            TEXT_CONTENT_TYPE,
            f"{self.command} is not supported; use {allowed}\n".encode(),
            {"Allow": ", ".join(ALLOWED_METHODS)},
        )

    def _respond(
        self,
        status: HTTPStatus,
        content_type: str,
        body: bytes,
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        """Send one response, writing the body only for a non-HEAD request."""
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for name, value in (extra_headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)


def _is_fixture_name(slug: str) -> bool:
    """Return whether `slug` names a fixture, not a path or a hidden file."""
    return bool(slug) and "/" not in slug and not slug.startswith(".")


def _make_server(root: Path, host: str, port: int) -> FixtureServer:
    """Bind, but do not start, a server serving the fixtures under `root`."""
    return FixtureServer((host, port), root)


def _origin(host: str, port: int) -> str:
    """Return the origin a client should use to reach `host` on `port`."""
    return f"http://{host}:{port}"


def serve(
    root: Path, port: int = 0, host: str = "127.0.0.1"
) -> tuple[FixtureServer, str]:
    """Serve `root` on a background thread, returning the server and its origin.

    `port=0` binds an ephemeral port, so several servers can run side by side;
    the one that was chosen is `server.server_port`. Point the pipeline at the
    returned origin, e.g. `update(data_dir=..., out_dir=..., base_url=origin)`,
    and stop the server with `server.shutdown()`.
    """
    server = _make_server(root, host, port)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, _origin(host, server.server_port)


def main() -> None:
    """Serve the fixtures in the foreground, logging where they are served."""
    parser = argparse.ArgumentParser(
        description="Serve recorded newspaper front pages to the pipeline."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="directory of recorded front pages",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",  # noqa: S104 (a container must be able to reach it)
        help="interface to bind",
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="port to bind; 0 picks a free one"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr
    )
    server = _make_server(args.root, args.host, args.port)
    logger.info("Serving %s on %s", server.root, _origin(args.host, server.server_port))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Stopping")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

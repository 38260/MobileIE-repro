"""Serve the repo to the tablet over the LAN and collect its benchmark results.

    python deploy/serve.py            # port 8770
    python deploy/serve.py 8899

It prints the single command to paste on the tablet. Results the tablet POSTs back
land in deploy/results_tablet/, so nothing has to be typed by hand.
"""
from __future__ import annotations

import http.server
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPLOADS = ROOT / "deploy" / "results_tablet"


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_POST(self):
        name = Path(self.path.split("?")[0]).name or "result.json"
        size = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(size)
        UPLOADS.mkdir(parents=True, exist_ok=True)
        (UPLOADS / name).write_bytes(body)
        print(f"\n[收到] deploy/results_tablet/{name}  ({len(body)} B)")
        print(body.decode("utf-8", "replace"))
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok\n")

    def log_message(self, fmt, *args):
        line = fmt % args
        if '" 200 ' in line or '" 304 ' in line:
            print("   ←", line.split('"')[0][-60:])


def lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8770
    ip = lan_ip()
    print(f"serving {ROOT}  on  http://{ip}:{port}/")
    print("在平板的 Ubuntu(proot) 里粘贴这一条：\n")
    print(f"    curl -s http://{ip}:{port}/deploy/bootstrap.sh | sh -s -- http://{ip}:{port}")
    print("\n（Ctrl+C 停止。跑完的结果会自动出现在 deploy/results_tablet/）")
    with http.server.ThreadingHTTPServer(("0.0.0.0", port), Handler) as httpd:
        httpd.serve_forever()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Push local HEAD commit to GitHub via the git Data API (bypasses git transport)."""
import base64
from pathlib import Path
import json
import subprocess
import sys
import urllib.request

REPO = "xueweijian/asr-wasm-speedlab"

def gh_token():
    out = subprocess.run(["git", "credential", "fill"],
                         input="protocol=https\nhost=github.com\n\n",
                         capture_output=True, text=True).stdout
    return [l.split("=", 1)[1] for l in out.splitlines() if l.startswith("password=")][0]

TOKEN = gh_token()

def api(path, data=None, method=None):
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(
        f"https://api.github.com/repos/{REPO}/{path}", data=body,
        method=method or ("POST" if body else "GET"),
        headers={"Authorization": f"token {TOKEN}",
                 "Accept": "application/vnd.github+json",
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        txt = r.read()
        return json.loads(txt) if txt else {}

def main():
    remote = api("git/ref/heads/main")["object"]["sha"]
    local = subprocess.run(["git", "rev-parse", "HEAD"],
                           capture_output=True, text=True).stdout.strip()
    if local == remote:
        print("already up to date")
        return
    # overlay ALL local commits since the last api-push (state file); remote
    # may have CI commits we cannot fetch — regenerated artifacts are identical
    state = Path(".push-state")
    base = state.read_text().strip() if state.exists() else None
    if base and subprocess.run(["git", "rev-parse", "--verify", "-q", base],
                               capture_output=True).returncode == 0:
        diffbase = base
    else:
        diffbase = subprocess.run(["git", "rev-list", "--max-parents=0", "HEAD"],
                                  capture_output=True, text=True).stdout.split()[0]
    files = subprocess.run(["git", "diff", "--name-only", "-z", diffbase, "HEAD"],
                           capture_output=True).stdout.decode().split("\0")
    files = [f for f in files if f]
    print(f"(diffbase {diffbase[:8]})")
    print(f"pushing {len(files)} files, HEAD {local[:8]} -> remote {remote[:8]}")

    base_tree = api(f"git/commits/{remote}")["tree"]["sha"]
    items = []
    for f in files:
        raw = subprocess.run(["git", "show", f"{local}:{f}"],
                             capture_output=True).stdout
        blob = api("git/blobs",
                   {"content": base64.b64encode(raw).decode(), "encoding": "base64"})
        items.append({"path": f, "mode": "100644", "type": "blob", "sha": blob["sha"]})
    tree = api("git/trees", {"base_tree": base_tree, "tree": items})

    msg = subprocess.run(["git", "log", "-1", "--pretty=%B", local],
                         capture_output=True, text=True).stdout.strip()
    commit = api("git/commits", {"message": msg + "\n(api-push)", "tree": tree["sha"],
                                 "parents": [remote]})
    api("git/refs/heads/main", {"sha": commit["sha"], "force": False}, "PATCH")
    Path(".push-state").write_text(local)
    print(f"pushed api-commit {commit['sha'][:8]}")

if __name__ == "__main__":
    main()

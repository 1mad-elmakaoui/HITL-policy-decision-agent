"""Report why the web interface is not rendering.

Prints the facts rather than guessing: which copy of the package Python is
importing, which file the route actually reads, and what the endpoint returns.

    python scripts/diagnose_web.py
"""

import sys
from pathlib import Path

# Running a script puts its own folder on sys.path, not the project root, so
# make the project importable the same way `python -m webapp` sees it.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

print("=" * 62)
print("  Policy Review Assistant - web interface diagnostic")
print("=" * 62)

# --- 1. which copy of the package is being imported? --------------------
# The usual cause of a page that looks correct on disk but not in the browser:
# the running server is importing a different copy of `webapp` (an installed
# one) whose static folder is empty or absent.
try:
    import webapp
except ImportError as exc:
    print(f"\nFAIL  cannot import webapp: {exc}")
    print("      Run from the project root, or: pip install -e .")
    sys.exit(1)

pkg = Path(webapp.__file__).parent
here = Path.cwd()
print(f"\n1. package imported from : {pkg}")
print(f"   current directory      : {here}")

if "site-packages" in str(pkg):
    print("   >> WARNING: importing an INSTALLED copy, not your working tree.")
    print("      Your edits are not being served. Fix with:")
    print("          pip uninstall -y policy-review-agent")
    print("          pip install -e \".[web]\"")
else:
    print("   OK  importing your working tree")

# --- 2. the file the route reads ---------------------------------------
from webapp.server import STATIC  # noqa: E402

page = STATIC / "index.html"
print(f"\n2. static dir            : {STATIC}")
print(f"   index.html            : {page}")

if not page.exists():
    print("   >> FAIL: that file does not exist.")
    sys.exit(1)

size = page.stat().st_size
raw = page.read_bytes()
print(f"   size                  : {size:,} bytes  (expected roughly 18,000-20,000)")

if size == 0:
    print("   >> FAIL: the file is empty.")
    sys.exit(1)

# A UTF-8 BOM or leading blank lines before <!doctype can stop some setups
# rendering the document as HTML.
if raw.startswith(b"\xef\xbb\xbf"):
    print("   >> WARNING: file starts with a UTF-8 BOM. Save it as UTF-8 without BOM.")

head = raw[:200].decode("utf-8", "replace").lstrip()
print(f"   starts with           : {head[:60]!r}")
if not head.lower().startswith("<!doctype"):
    print("   >> WARNING: does not start with <!doctype html>")

if b"Policy Review Assistant" not in raw:
    print("   >> WARNING: expected page title not found - is this the right file?")
if b"--bg:#0b0f1a" not in raw:
    print("   >> WARNING: the stylesheet is missing. A white page means no CSS loaded.")

# --- 3. what the endpoint returns --------------------------------------
print("\n3. calling the route in-process")
try:
    from fastapi.testclient import TestClient

    from webapp.server import app

    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/")
    body = response.content
    print(f"   GET /                 : HTTP {response.status_code}, {len(body):,} bytes")
    print(f"   content-type          : {response.headers.get('content-type')}")

    if response.status_code != 200:
        print(f"   >> FAIL: {body[:300].decode('utf-8','replace')}")
    elif len(body) == 0:
        print("   >> FAIL: 200 with an empty body - a browser renders this blank and white.")
    else:
        print("   OK  the route serves the page correctly")

    status = client.get("/api/status")
    print(f"   GET /api/status       : HTTP {status.status_code}")
    if status.status_code == 200:
        data = status.json()
        print(f"   run mode              : {data['run_mode']}")
        index = data.get("index")
        print(f"   index                 : "
              + (f"{index['chunk_count']} chunks" if index else f"UNAVAILABLE - {data.get('index_error')}"))
except Exception as exc:  # noqa: BLE001
    print(f"   could not call the app in-process: {exc}")

print("\n" + "=" * 62)
print("If everything above says OK but the browser is still blank, the browser")
print("cached the earlier empty response. Hard-reload with Ctrl+Shift+R, or")
print("open http://127.0.0.1:8000 in a private window.")
print("=" * 62)
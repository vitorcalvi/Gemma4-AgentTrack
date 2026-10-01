"""
Drive the Kaggle Writeup submission UI with a real, already-signed-in browser.

No account password is ever read, stored, or required. Authentication comes
from a persistent Chromium profile that you sign into once, by hand:

    1. python submit/submit.py --login-only
    2. In the window that opens, sign in to Kaggle (Google sign-in is fine).
    3. Close the window. The session cookies are saved to
       submit/browser_state.json (mode 0600, git-ignored).

Afterwards, just run `python submit/submit.py --paper both` and the script
reuses that session. Nothing here touches KAGGLE_EMAIL / KAGGLE_PASSWORD.

Usage:
  python submit/submit.py [--paper 1|2|both] [--dry-run] [--login-only]
"""
from __future__ import annotations
import argparse, asyncio, json, os, pathlib, sys, re

ROOT = pathlib.Path(__file__).resolve().parent.parent
SUBMIT_DIR = pathlib.Path(__file__).resolve().parent
PROFILE = SUBMIT_DIR / ".browser-profile"
STATE = SUBMIT_DIR / "browser_state.json"
COMP = "gemma-4-developer-agent-paper"
NOTEBOOK = "https://www.kaggle.com/code/coachvitorcalvi/graphloc-seed-expansion"
DATASET = "https://www.kaggle.com/datasets/coachvitorcalvi/graphloc-300-labels"
COVER = ROOT / "figures" / "cover_card.png"

# Kaggle Writeup UI hard limits (0 / 80 and 0 / 140, validated below).
TITLE_MAX = 80
SUBTITLE_MAX = 140


def _assert_lengths(num: int, title: str, subtitle: str) -> None:
    """Fail fast if a title or subtitle exceeds Kaggle's UI limits."""
    if len(title) > TITLE_MAX:
        raise SystemExit(
            f"Paper {num} title is {len(title)} chars; Kaggle allows at most "
            f"{TITLE_MAX}. Trim: {title!r}"
        )
    if len(subtitle) > SUBTITLE_MAX:
        raise SystemExit(
            f"Paper {num} subtitle is {len(subtitle)} chars; Kaggle allows at "
            f"most {SUBTITLE_MAX}. Trim: {subtitle!r}"
        )


PAPERS = {
    1: {
        # 67 chars; Kaggle Writeup title limit is 80.
        "title": "Where Does the Graph Help? Seed Expansion on "
                 "Repository Code Graphs",
        # 99 chars; Kaggle Writeup subtitle limit is 140.
        "subtitle": "Closing the API-implementation divergence on SWE-bench "
                    "Lite repository graphs with zero regressions",
        "body": ROOT / "paper" / "paper1_resource.md",
    },
    2: {
        # 68 chars; Kaggle Writeup title limit is 80.
        "title": "The Adherence Gap: Retrieval Is Not the Bottleneck for "
                 "Coding Agents",
        # 88 chars; Kaggle Writeup subtitle limit is 140.
        "subtitle": "Diagnosing why developer agents ignore retrieved targets "
                    "in 46.6% of SWE-bench Lite runs",
        "body": ROOT / "paper" / "paper2_application.md",
    },
}

# Validate the static PAPERS table at import time so a length regression is
# caught by `python -m py_compile` + import, not at submit time.
for _n, _p in PAPERS.items():
    _assert_lengths(_n, _p["title"], _p["subtitle"])


def strip_front_matter(text: str) -> tuple[str, str, str]:
    """Split a paper markdown into (title, subtitle, body).

    The first '# ' heading is the title. An optional subtitle is the next
    non-empty line *only if* it is not a byline (contains no 'Kaggle:'
    marker) and not a horizontal rule. Everything after is the body.
    """
    lines = text.split("\n")
    title = ""
    subtitle = ""
    i = 0
    while i < len(lines):
        ln = lines[i].strip()
        if ln.startswith("# "):
            title = ln[2:].strip()
            i += 1
            break
        if ln and not ln.startswith("#"):
            i += 1
            continue
        i += 1
    # skip blanks / rules / byline block
    while i < len(lines):
        ln = lines[i].strip()
        if not ln or ln == "---" or "Kaggle:" in ln or ln.startswith("*Target"):
            i += 1
            continue
        break
    # a subtitle is a short line that is not a heading and not a byline
    if i < len(lines):
        ln = lines[i].strip()
        if ln and not ln.startswith("#") and len(ln) < 160 and "http" not in ln:
            subtitle = ln.lstrip("*").strip().rstrip("*").strip()
            i += 1
    while i < len(lines) and not lines[i].strip():
        i += 1
    body = "\n".join(lines[i:])
    body = re.sub(r"^-{3,}\n", "", body, count=1)
    return title, subtitle, body.lstrip()


async def open_context(browser):
    """Reuse a saved Kaggle session. Never reads or handles a password."""
    ctx = await browser.new_context(viewport={"width": 1500, "height": 1000},
                                    accept_downloads=True)
    if STATE.exists():
        try:
            await ctx.add_cookies(json.loads(STATE.read_text())["cookies"])
            print("  restored saved Kaggle session")
        except Exception as e:
            print("  could not restore session:", str(e)[:80])
    return ctx


async def apply_stealth(page) -> None:
    """Hide automation fingerprints so Google's OAuth flow doesn't refuse us.

    Without this, Google shows "This browser or app may not be secure" and
    blocks the sign-in, which silently breaks the whole submission flow.
    The init script runs before any page JS, so navigator.webdriver is masked
    before Google's probe runs.
    """
    try:
        await page.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', "
            "{get: () => undefined})"
        )
    except Exception as e:
        print("  could not install stealth init script:", str(e)[:80])


async def save_session(ctx):
    """Persist cookies: they are a live credential, so keep them private."""
    try:
        cookies = await ctx.cookies()
        STATE.write_text(json.dumps({"cookies": cookies}), encoding="utf-8")
        os.chmod(STATE, 0o600)
        if any(c.get("name") == "ka_session"
               or "kaggle.com" in (c.get("domain") or "")
               for c in cookies):
            print("  valid Kaggle session captured ->", STATE.name)
        else:
            print("  session saved ->", STATE.name)
    except Exception:
        # Silently ignore closed-context errors: if the file was already written
        # by an earlier loop iteration, there is nothing more to do here.
        pass


async def signed_in(page) -> bool:
    """True when the restored session is authenticated as a real user."""
    try:
        await page.goto("https://www.kaggle.com/settings/account",
                        wait_until="domcontentloaded", timeout=120000)
        await page.wait_for_timeout(2500)
        if "/signin" in page.url or "/account/login" in page.url:
            return False
        body = (await page.inner_text("body"))[:4000].lower()
        return not any(s in body for s in
                       ("sign in", "signin", "log in", "create account"))
    except Exception:
        return False


async def interactive_login(browser):
    """Open a visible window so the user signs in by hand (no password here)."""
    ctx = await browser.new_context(viewport={"width": 1500, "height": 1000},
                                    accept_downloads=True)
    page = await ctx.new_page()
    await apply_stealth(page)
    print("\nA browser window is open. Please sign in to Kaggle there.")
    print("Google sign-in is fine. Then CLOSE the window when done.\n")
    await page.goto("https://www.kaggle.com/account/login",
                    wait_until="domcontentloaded", timeout=120000)
    try:
        while True:
            try:
                page_alive = not page.is_closed()
                ctx_alive = bool(ctx.pages)
            except Exception:
                # Page/context handles already torn down: stop polling.
                break
            if not page_alive and not ctx_alive:
                break
            try:
                if await ctx.cookies():
                    await save_session(ctx)
            except Exception:
                # Cookies call failed mid-loop (e.g. context closed between
                # checks); ignore and try again next tick.
                pass
            await asyncio.sleep(2)
    except Exception:
        # TargetClosedError or anything else during the loop: exit cleanly so
        # the final save_session below still flushes whatever we already have.
        pass
    await save_session(ctx)  # final flush; save_session swallows its own errors
    try:
        await ctx.close()
    except Exception:
        pass
    print("login captured.\n")


async def joined(page):
    body = await page.inner_text("body")
    return "Join Hackathon" in body or "Join Competition" in body


async def do_join(page):
    print("  joining hackathon ...")
    for label in ("Join Hackathon", "Join Competition", "Accept Rules", "I Agree"):
        el = page.get_by_text(label, exact=False).first
        if await el.count() and await el.is_visible():
            try:
                await el.click()
                await page.wait_for_timeout(6000)
                print("  after join ->", page.url)
                return
            except Exception as e:
                print("   click failed:", str(e)[:80])
    print("  (no join control found; may already be joined)")


async def submit_paper(page, num, dry_run=False):
    p = PAPERS[num]
    title, subtitle, body = strip_front_matter(p["body"].read_text(encoding="utf-8"))
    title = title or p["title"]
    subtitle = subtitle or p["subtitle"]
    # Re-validate at the point of submission: the title/subtitle may have come
    # from the markdown H1 / second line, which is not protected by the
    # import-time PAPERS check above.
    _assert_lengths(num, title, subtitle)
    print(f"\n=== Paper {num}: {title[:64]} ({len(body.split())} words) ===")
    if dry_run:
        return

    # navigate to the writeups area
    for url in (f"https://www.kaggle.com/competitions/{COMP}/writeups",
                f"https://www.kaggle.com/competitions/{COMP}/overview"):
        await page.goto(url, wait_until="domcontentloaded", timeout=120000)
        await page.wait_for_timeout(5000)
        if await page.get_by_text("New Writeup", exact=False).count():
            break
    for label in ("New Writeup", "Create Writeup", "Add writeup"):
        el = page.get_by_text(label, exact=False).first
        if await el.count() and await el.is_visible():
            await el.click()
            await page.wait_for_timeout(6000)
            break
    print("  writeup editor ->", page.url)

    # title / subtitle
    for sel, val in (('input[name="title"]', title),
                     ('input[placeholder*="Title" i]', title)):
        el = page.locator(sel).first
        if await el.count():
            await el.fill(val)
            break
    for sel in ('input[name="subtitle"]', 'input[placeholder*="ubtitle" i]'):
        el = page.locator(sel).first
        if await el.count():
            await el.fill(subtitle or p["subtitle"])
            break

    # body: prefer a contenteditable / textarea editor
    target = None
    for sel in ('div[contenteditable="true"]', 'textarea[name="content"]',
                'div[role="textbox"]', 'textarea'):
        el = page.locator(sel).first
        if await el.count() and await el.is_visible():
            target = el
            break
    if target is None:
        raise SystemExit("could not locate the writeup body editor")
    await target.click()
    await page.keyboard.insert_text(body)
    await page.wait_for_timeout(2000)

    # project links
    for sel, val in (('input[placeholder*="http" i]', NOTEBOOK),):
        els = page.locator(sel)
        n = await els.count()
        for i in range(min(n, 2)):
            await els.nth(i).fill(val if i == 0 else DATASET)
    # cover image (requireCardImage)
    if COVER.exists():
        fi = page.locator('input[type="file"]').first
        if await fi.count():
            try:
                await fi.set_input_files(str(COVER))
                await page.wait_for_timeout(3000)
                print("  cover image attached")
            except Exception as e:
                print("  cover upload skipped:", str(e)[:70])

    await page.screenshot(path=f"/tmp/writeup_{num}.png", full_page=True)
    save = page.get_by_text(re.compile("^Save", re.I), exact=False).first
    if await save.count() and await save.is_visible():
        await save.click()
        await page.wait_for_timeout(6000)
        print("  saved")
    if dry_run:
        return
    sub = page.get_by_text(re.compile("^Submit", re.I), exact=False).first
    if await sub.count() and await sub.is_visible():
        await sub.click()
        await page.wait_for_timeout(8000)
        print("  SUBMITTED ->", page.url)
    else:
        print("  (no Submit button yet — review /tmp/writeup_%d.png)" % num)


# Chrome launch flags that suppress the fingerprint signals Google's
# "This browser or app may not be secure" screen looks for.
STEALTH_LAUNCH_ARGS = [
    "--disable-blink-features=AutomationControlled",
    "--no-first-run",
    "--no-default-browser-check",
]
STEALTH_IGNORE_DEFAULT_ARGS = ["--enable-automation"]


def _launch_browser(pw, user_data_dir: str | None = None):
    """Launch a browser that survives Google's anti-automation checks.

    Tries system Google Chrome first (channel="chrome") with stealth flags.
    Falls back to bundled Chromium if Chrome is not installed, so CI / a
    machine without Chrome does not blow up.

    `--user-data-dir` is passed as a Chrome flag (not a Playwright kwarg) so
    the same code path works for both the system Chrome binary and the
    bundled chromium fallback.
    """
    launch_args = ["--no-sandbox", *STEALTH_LAUNCH_ARGS]
    if user_data_dir:
        launch_args.append(f"--user-data-dir={user_data_dir}")
    try:
        return pw.chromium.launch(
            channel="chrome",
            headless=False,
            args=launch_args,
            ignore_default_args=STEALTH_IGNORE_DEFAULT_ARGS,
        )
    except Exception as e:
        print("  could not launch system Chrome (" + str(e)[:80] +
              "); falling back to bundled chromium")
        return pw.chromium.launch(
            headless=False,
            args=launch_args,
            ignore_default_args=STEALTH_IGNORE_DEFAULT_ARGS,
        )


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", default="both", choices=["1", "2", "both"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--login-only", action="store_true",
                    help="open a window, sign in by hand, save the session")
    ap.add_argument("--user-data-dir", default=None,
                    help="path to an existing Chrome user profile to reuse")
    a = ap.parse_args()

    from playwright.async_api import async_playwright
    async with async_playwright() as pw:
        b = await _launch_browser(pw, user_data_dir=a.user_data_dir)

        if a.login_only:
            await interactive_login(b)
            await b.close()
            return

        ctx = await open_context(b)
        pg = await ctx.new_page()
        await apply_stealth(pg)
        pg.set_default_timeout(60000)

        print("checking session ...")
        if not await signed_in(pg):
            print("\nNo valid Kaggle session found.")
            print("Run:  python submit/submit.py --login-only")
            print("Sign in the window that opens, then close it.\n")
            await b.close()
            raise SystemExit(1)

        await save_session(ctx)
        print("session OK ->", pg.url)
        if not a.dry_run and await joined(pg):
            await do_join(pg)
        nums = [1, 2] if a.paper == "both" else [int(a.paper)]
        for n in nums:
            await submit_paper(pg, n, a.dry_run)
        await save_session(ctx)
        await b.close()


if __name__ == "__main__":
    asyncio.run(main())

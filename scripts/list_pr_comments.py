"""List top-level GitHub comments for a pull request.

Usage:
    uv run python scripts/list_pr_comments.py --pr https://github.com/OWNER/REPO/pull/123
"""

from __future__ import annotations

import argparse
import sys

import httpx
from dotenv import load_dotenv

from common.github import API, _headers, parse_pr_url


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--pr", required=True, help="GitHub pull request URL")
    args = parser.parse_args()

    owner, repo, number = parse_pr_url(args.pr)
    url = f"{API}/repos/{owner}/{repo}/issues/{number}/comments"
    comments = []
    with httpx.Client(timeout=30.0) as client:
        page = 1
        while True:
            resp = client.get(url, headers=_headers(), params={"per_page": 100, "page": page})
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break
            comments.extend(batch)
            if len(batch) < 100:
                break
            page += 1

    for comment in comments:
        body = " ".join((comment.get("body") or "").split())
        preview = body[:200]
        if len(body) > 200:
            preview += "..."
        print(f"{comment['user']['login']} | {comment['created_at']} | {preview}")


if __name__ == "__main__":
    main()

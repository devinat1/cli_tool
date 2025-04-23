"""
changelog_cli.py: Generate a user-friendly changelog for the last N git commits
using the OpenAI API

Usage:
    python changelog_cli.py N [--api-key KEY] [--branch BRANCH] [--repo REPO] [--markdown]

Example:
    python changelog_cli.py 10 --api-key sk-abc123 --repo /path/to/repo --markdown
Testing:
    pytest main.py
"""
import argparse
import sys
import asyncio
import json
import aiohttp
import logging
import re
import os
from abc import ABC, abstractmethod
from typing import Any, Optional
from dotenv import load_dotenv

load_dotenv()

# Set up basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class Config:
    """Manage API configuration and headers."""

    DEFAULT_API_KEY = None
    BASE_URL = "https://api.openai.com"
    API_ENDPOINT = "/v1/chat/completions"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

    @property
    def full_url(self) -> str:
        """Return the full endpoint URL."""
        return f"{self.BASE_URL}{self.API_ENDPOINT}"

    @property
    def headers(self) -> dict[str, str]:
        """Return HTTP headers for requests."""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }


class GitCommitFetcher:
    """Fetch the last N commits from a local or remote git repository."""

    def __init__(self, repo_path: Optional[str] = None):
        self.repo_path = repo_path
        self.is_remote = bool(repo_path and self._is_remote_repo(repo_path))

    @staticmethod
    def _is_remote_repo(repo_path: str) -> bool:
        """Check if the repository path refers to a remote URL."""
        return (
            repo_path.startswith("http://")
            or repo_path.startswith("https://")
            or repo_path.endswith(".git")
            or repo_path.startswith("git@")
        )

    async def get_last_commits(self, n: int, branch: Optional[str] = None) -> str:
        """
        Retrieve a plain-text dump of the last N commits (hash, subject, body).
        Supports GitHub REST API for remote GitHub repos; otherwise uses local git.
        """
        if self.is_remote and "github.com" in (self.repo_path or ""):
            return await self._fetch_commits_github_api(n, branch or "main")

        base_cmd = ["git", "log", f"-n{n}"]
        if self.is_remote:
            # Use SSH for remote git access
            base_cmd[0:0] = [
                "git",
                "-c",
                "core.sshCommand=ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new",
            ]
            remote_branch = branch or "main"
            base_cmd.append(f"{self.repo_path}:{remote_branch}")
        elif branch:
            base_cmd.append(branch)

        base_cmd.append("--pretty=format:%H%n%s%n%b%n----END_COMMIT----")

        proc = await asyncio.create_subprocess_exec(
            *base_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=None if self.is_remote else self.repo_path,
        )
        out, err = await proc.communicate()
        if proc.returncode != 0:
            logger.error("git log failed: %s", err.decode(errors="replace"))
            sys.exit(1)

        return out.decode("utf-8", errors="replace")

    async def _fetch_commits_github_api(self, n: int, branch: str) -> str:
        """Fetch commits from GitHub REST API for a remote repository."""
        url = self.repo_path.rstrip(".git")
        owner, repo = url.split("/")[-2:]
        api_url = f"https://api.github.com/repos/{owner}/{repo}/commits"
        async with aiohttp.ClientSession() as session:
            async with session.get(
                api_url, params={"sha": branch, "per_page": n}
            ) as resp:
                if resp.status != 200:
                    logger.error(
                        "GitHub API error %s: %s", resp.status, await resp.text()
                    )
                    sys.exit(1)
                data = await resp.json()

        lines: list[str] = []
        for commit in data:
            sha = commit["sha"]
            msg = commit["commit"]["message"]
            subject, *rest = msg.split("\n", 1)
            body = rest[0] if rest else ""
            lines.extend([sha, subject, body or "", "----END_COMMIT----"])

        return "\n".join(lines)


class ContextWindowError(Exception):
    """Raised when the API context window is exceeded."""
    pass


class ApiClientInterface(ABC):
    """Interface for API clients."""

    @abstractmethod
    async def send_request(self, payload: dict[str, Any]) -> Any:
        pass


class OpenAIApiClient(ApiClientInterface):
    """Client for communicating with the OpenAI API."""

    def __init__(self, config: Config):
        self.config = config

    async def send_request(self, payload: dict[str, Any]) -> Any:
        """Send a JSON payload to the API and return the response."""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.config.full_url, headers=self.config.headers, json=payload
            ) as resp:
                if resp.status == 400:
                    raise ContextWindowError(
                        "Context window limit exceeded. Reduce the number of commits."
                    )
                if resp.status != 200:
                    logger.error("API error %s: %s", resp.status, await resp.text())
                    sys.exit(1)
                return await resp.json()


class ChangelogGenerator:
    """Generate a formatted changelog from git commit text."""

    def __init__(self, api_client: ApiClientInterface):
        self.api_client = api_client

    async def generate_changelog(self, git_log: str) -> str:
        """
        Send the git log to the API and return the generated changelog text.
        """
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Based on the following git log, generate a user‑friendly changelog formatted like professional release notes. "
                        "Use clear section headings (e.g., Improvements, Bug fixes) and bullet points. "
                        "Follow this example style:\n\n"
                        "Release Notes Title\n\n"
                        "Improvements\n"
                        "* Tag changelog updates so end users can filter updates\n"
                        "* Sonnet-3.7 supported for AI Chat. Configure your preferred model through the dashboard\n"
                        "* Change your deployment name directly in dashboard settings\n\n"
                        "Bug fixes\n"
                        "* OG images fixed\n"
                        "* Fixed icon style inconsistency for anchors without container\n"
                        "* Improved styling nits for dashboard borders on mobile/tablet\n"
                        "* Show code examples even in simple mode for API playground\n"
                        "* Support “command + k” shortcut for search in web editor\n"
                        "* Codeblocks within callouts expand to fill the width of the callout area\n\n"
                        "Now generate a changelog using the same structure for this git log. Include just the changelog in your output. :\n\n"
                        f"{git_log}"
                    ),
                }
            ],
            "max_tokens": 4096,
            "temperature": 0.5,
        }
        return await self.api_client.send_request(payload)


class CliParser:
    """Parse command-line arguments."""

    @staticmethod
    def parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description="Generate a changelog from the last N git commits via the OpenAI API"
        )
        parser.add_argument("n", type=int, help="Number of commits to include")
        parser.add_argument(
            "--api-key",
            "-k",
            help="Override the default API key",
        )
        parser.add_argument(
            "--branch",
            "-b",
            default="main",
            help="Branch to fetch commits from (default: main)",
        )
        parser.add_argument(
            "--repo",
            "-r",
            default=None,
            help="Path or URL of the git repository (default: current directory)",
        )
        parser.add_argument(
            "--markdown",
            "-m",
            action="store_true",
            help="Save the generated changelog to changelog.md",
        )
        return parser.parse_args()


class ChangelogApp:
    """Orchestrate fetching commits, generating, and outputting the changelog."""

    def __init__(self) -> None:
        self.args = CliParser.parse_args()
        self.config = Config(api_key=self.args.api_key)
        self.api_client = OpenAIApiClient(self.config)
        self.generator = ChangelogGenerator(self.api_client)
        self.commit_fetcher = GitCommitFetcher(repo_path=self.args.repo)

    async def run(self) -> None:
        """Execute the application workflow."""
        if self.args.n <= 0:
            logger.error("Number of commits must be positive; got %d", self.args.n)
            sys.exit(1)

        git_log = await self.commit_fetcher.get_last_commits(
            self.args.n, branch=self.args.branch
        )

        actual = git_log.count("----END_COMMIT----")
        if actual < self.args.n:
            logger.warning(
                "Requested %d commits but only %d available.", self.args.n, actual
            )

        MAX_LOG_CHARS = 15000
        if len(git_log) > MAX_LOG_CHARS:
            logger.warning("Git log too large; truncating for context window.")
            git_log = git_log[-MAX_LOG_CHARS:]

        try:
            result = await self.generator.generate_changelog(git_log)
        except ContextWindowError as exc:
            logger.error(str(exc))
            sys.exit(1)

        text = self._extract_text(result)
        self._validate_changelog(text)
        print(text)

        if self.args.markdown:
            with open("changelog.md", "w") as md_file:
                md_file.write(text)

    @staticmethod
    def _extract_text(response: Any) -> str:
        """Extract the generated text from the API response."""
        if isinstance(response, dict):
            if "completion" in response:
                return response["completion"].strip()
            if "choices" in response and response["choices"]:
                return (
                    response["choices"][0].get("message", {}).get("content", "").strip()
                )
            if "content" in response and isinstance(response["content"], list):
                return "".join(
                    chunk.get("text", "") for chunk in response["content"]
                ).strip()
            return json.dumps(response, indent=2)
        return str(response)

    @staticmethod
    def _validate_changelog(text: str) -> None:
        """Ensure the generated changelog is not empty and has bullet points."""
        if not text.strip():
            logger.error("Changelog validation failed: output is empty.")
            sys.exit(1)
        if not re.search(r'^\* ', text, re.MULTILINE):
            logger.error("Changelog validation failed: no bullet points found.")
            sys.exit(1)


def main() -> None:
    """Main entry point."""
    app = ChangelogApp()
    asyncio.run(app.run())


if __name__ == "__main__":
    main()


# -------------------------- TESTS BELOW --------------------------
import pytest

# Config tests
def test_config_full_url_and_headers():
    cfg = Config(api_key="KEY123")
    assert cfg.full_url.endswith("/v1/chat/completions")
    assert cfg.headers["Authorization"] == "Bearer KEY123"
    assert cfg.headers["Content-Type"] == "application/json"

# GitCommitFetcher._is_remote_repo
@pytest.mark.parametrize("path,expected", [
    ("http://example.com/repo.git", True),
    ("https://git.com/x/y.git", True),
    ("git@github.com:user/repo.git", True),
    ("/local/path", False),
    ("relative/path", False),
])
def test_is_remote_repo(path, expected):
    assert GitCommitFetcher._is_remote_repo(path) is expected

# CliParser.parse_args
def test_parse_args(monkeypatch):
    monkeypatch.setenv("PYTEST_RUNNING", "1")
    monkeypatch.setattr(sys, "argv", ["prog", "5", "-k", "ABC", "-b", "dev", "-r", "/tmp", "-m"]
)
    args = CliParser.parse_args()
    assert args.n == 5
    assert args.api_key == "ABC"
    assert args.branch == "dev"
    assert args.repo == "/tmp"
    assert args.markdown is True

# ChangelogGenerator payload and return
def test_generate_changelog_payload_and_return():
    class DummyClient:
        def __init__(self):
            self.payload = None
        async def send_request(self, payload):
            self.payload = payload
            return {"completion": "OK"}

    dummy = DummyClient()
    gen = ChangelogGenerator(dummy)
    result = asyncio.get_event_loop().run_until_complete(gen.generate_changelog("LOGDATA"))
    assert result == {"completion": "OK"}
    p = dummy.payload
    assert "model" in p and p["model"].startswith("gpt")
    msg = p["messages"][0]["content"]
    assert "LOGDATA" in msg

# ChangelogApp text extraction
@pytest.mark.parametrize("response,expected", [
    ({"completion": " TEXT "}, "TEXT"),
    ({"choices": [{"message": {"content": "  Hello\n"}}]}, "Hello"),
    ({"content": [{"text": "A"}, {"text": "B"}]}, "AB"),
    ({"foo": "bar"}, json.dumps({"foo": "bar"}, indent=2)),
    ("plain", "plain"),
])
def test_extract_text(response, expected):
    got = ChangelogApp._extract_text(response)
    assert got == expected

# ChangelogApp validation
def test_validate_changelog_empty():
    with pytest.raises(SystemExit):
        ChangelogApp._validate_changelog("   ")

def test_validate_changelog_no_bullets():
    with pytest.raises(SystemExit):
        ChangelogApp._validate_changelog("No bullets here")

def test_validate_changelog_ok():
    # should not raise
    ChangelogApp._validate_changelog("* item\n* another")

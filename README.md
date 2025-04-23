# Changelog CLI Tool

Generate a user‑friendly changelog for the last N git commits using OpenAI

## Prerequisites

- Python 3.7 or newer
- Git
- Internet access (for remote repos or API calls)

## Installation

1. Clone the repository:
   ```bash
   git clone /path/to/your/repo
   cd cli_tool
   ```
2. (Optional) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

```bash
python changelog_cli.py N [options]
```

Where `N` is the number of commits to include.

Options:
- `-k, --api-key KEY`  
  Override the default API key.
- `-b, --branch BRANCH`  
  Branch to fetch commits from (default: `main`).
- `-r, --repo REPO`  
  Path or URL of the git repository (default: current directory).
- `-m, --markdown`  
  Save the generated changelog to `changelog.md`.

### Examples

Generate a changelog for the last 5 commits in the current repo:
```bash
python changelog_cli.py 5
```

Specify a branch and save as markdown:
```bash
python changelog_cli.py 10 -b develop -m
```

Use a custom API key:
```bash
python changelog_cli.py 8 -k sk-abc123
```

Fetch from a remote GitHub repository:
```bash
python changelog_cli.py 6 -r https://github.com/user/repo.git
```

## Testing

Run the built‑in tests with pytest:
```bash
python3 -m pytest changelog_cli.py
```

## Note
Used GitHub Copilot to aid with feature development

## Website

You can view the changelog as a static website:

1. Generate or update `changelog.md` with your latest commits:
   ```bash
   python changelog_cli.py N -m
   ```
2. Open `site/index.html` in your browser.

The page will fetch `changelog.md` and render it via a Markdown parser.

### Technical & Product Decisions

I chose a plain‑HTML frontend to keep the experience as simple and lightweight as possible—no build step or server required. While I’m comfortable with React and Next.js, this static HTML + marked.js approach delivers instant load, easy deployment, and minimal maintenance.

## Design Rationale

- We kept as few files as possible to make it easier to traverse the repo and locate key functionality quickly.

Questions:
- **Does it work?**  
  Yes – commands run out‑of‑the‑box, generate changelogs, and render correctly in both CLI and static site modes.
- **Does the backend logic make sense?**  
  The fetch‑generate‑output flow is straightforward: retrieve commits, send to API, validate and print or save.
- **Is there evidence of user‑centered product choices?**  
  We’ve prioritized simplicity (no server, minimal deps), clear defaults, and helpful warnings to guide users.
- **Is it pretty (simple and minimal can be beautiful)?**  
  The UI is clean: one HTML file, a basic CSS, and Markdown styling. It’s intentionally minimal yet readable.
- **How is the UX from the developer’s perspective?**  
  Very easy – install deps, run one script, and get a polished changelog. No build steps, no config complexity.

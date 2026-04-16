# `lit-review/` — HippoCortex literature-review vault

This folder is an [Obsidian](https://obsidian.md) vault. Open the `lit-review/` folder as a vault to get backlinks, graph view, and tag search across every note.

## Folders

| Folder       | What goes here                                                                          |
| ------------ | --------------------------------------------------------------------------------------- |
| `papers/`    | One note per paper. Start from `_template.md`. The 9 proposal references are pre-seeded.|
| `topics/`    | Concept notes (continual learning, SSMs, SWRs, generative replay, null-space projection). Link papers in from `papers/`. |
| `meetings/`  | Supervisor + team meeting notes. Start from `_template.md`.                             |
| `daily/`     | Daily research log. Start from `_template.md`.                                          |

## Conventions

- **Paper note filename**: `<firstauthor><year>-<slug>.md` (e.g. `gu2023-mamba.md`).
- **Tags** go in YAML frontmatter: `tags: [continual-learning, ssm, generative-replay]`.
- **Status field** on paper notes: `to-read` / `reading` / `done`. Lets Obsidian Dataview filter a reading queue.
- **Link generously** — every concept note should backlink to the papers that define or use it.

## Suggested Obsidian plugins

Not configured yet — install once a team member opens the vault for the first time:

- **Dataview** — query paper notes by tag / status / year.
- **Citations** (or **Zotero Integration**) — pull BibTeX straight into notes.
- **Templater** — auto-fill the paper / meeting / daily templates.

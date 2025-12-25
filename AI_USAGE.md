# AI Tool Usage Disclosure (AI_USAGE.md)

## Tools used
- **ChatGPT**: used as an assistant for debugging, improvement suggestions (pipeline, time-series validation, dashboard), and occasionally for proposing code examples.
- **GitHub Copilot**: used mainly for auto-completion (code suggestions) and small syntax adjustments.

## Nature of the AI assistance (what the AI did)
AI was used **occasionally**, mainly:
- when I encountered a **bug** (imports, environment setup, execution, tests),
- when I was unsure how to **structure/improve** the pipeline (script organization, CLI options, reproducible outputs),
- when I wanted to **improve results** (baselines, temporal validation, “post-disaster years” diagnostics),
- to propose **dashboard** ideas to better present and analyze results,
- and **sometimes** to generate **small code snippets** (e.g., utility functions, parts of scripts, targeted fixes).

## What I did myself
- I integrated the code into my project, **adapted** the suggestions to my architecture and data, and made the final decisions (models, features, validation).
- I reviewed and modified the suggested portions to ensure they match the course requirements.
- I checked data consistency and avoided look-ahead bias (strict vs ex-post modes).
- I validated correctness and reproducibility by running the project and the test suite.

## Where AI may have influenced the work (relevant files)
- `main.py`
- `src/models.py`
- `src/features.py`
- `src/data_loading.py`
- `script/dashboard.py` (and/or `dashboard/build_dashboard.py` if present)
- `tests/test_features.py`, `tests/test_pipeline.py`, `tests/test_models.py`

## What the AI did NOT do
- The AI did not “write the entire project”: it provided suggestions, examples, and snippets that I then **selected**, **integrated**, **adapted**, and **tested**.
- The AI tools did not execute code on my machine or directly access my files.
- I did not submit code that I do not understand.

## Responsibility
I take full responsibility for the final code, results, and interpretation. AI tools (ChatGPT and Copilot) were used as occasional assistance (debugging, suggestions, auto-completion, and code snippets), but all final decisions and validations were made by me.

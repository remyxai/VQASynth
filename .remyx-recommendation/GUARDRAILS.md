# Path guardrails for this PR

You MAY create files matching:
```
vqasynth/*_integration.py
tests/test_*_integration.py
tests/test_*.py
.remyx-recommendation/**
.feature-finder/**
README.md
```

You MAY append-only modify:
```
README.md
```

You MUST NOT touch:
```
.github/**
docker/**
pipelines/**
config/**
requirements.txt
setup.py
pyproject.toml
MANIFEST.in
```

After the orchestrator validates your work, it checks the diff with
`git diff --name-only`. If any path you touched is outside the allowed
set, the PR is rejected and your work is not committed.

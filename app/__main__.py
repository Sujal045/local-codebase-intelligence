"""Allow ``python -m app`` as an alias for the CLI."""

from app.cli import main

raise SystemExit(main())

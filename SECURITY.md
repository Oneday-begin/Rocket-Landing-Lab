# Secret handling

No API key is required by the local application or public phone demo. Local Ollama supplies optional language generation. Never commit passwords, provider keys, private keys, `.env` files or credential stores. Browser assets and the public repository are readable by visitors.

GitHub secret scanning and secret-scanning push protection are enabled. The `deployment/` publisher additionally requires Gitleaks and a strict export guard before pushing to `gh-pages`; see its README and the final chapter of the placement report. The audited baseline on 25 October 2026 contained two branches and five reachable commits, with no detected leaks. Clean scans are evidence within the checked scope, not a guarantee against every possible secret.

If an actual credential is exposed, revoke or rotate it with its provider before arranging history cleanup. Deleting a current file or adding an ignore rule does not remove historical exposure. Do not post a discovered secret in an issue, screenshot or chat.

"""Test-session setup.

litellm calls `load_dotenv()` the first time it is imported unless LITELLM_MODE is PRODUCTION. That
would copy a developer's real `.env` (API keys included) into the test process, and make tests
that expect a missing key depend on which test happened to import litellm first. Setting the mode
here, before any test module can import litellm, keeps the suite independent of any local `.env`.
"""

import os

os.environ.setdefault("LITELLM_MODE", "PRODUCTION")

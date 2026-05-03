"""BallotBox AI — REST API module.

Defines FastAPI routes, request/response models, and middleware
for the election literacy platform.
"""

from api.routes import router

__all__ = ["router"]

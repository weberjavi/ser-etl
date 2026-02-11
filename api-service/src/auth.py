"""Auth0 JWT authentication."""

import jwt
from jwt.exceptions import InvalidTokenError
import requests
from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Dict, Any
import structlog

from .config import settings

logger = structlog.get_logger()
security = HTTPBearer()


class Auth0Validator:
    """Validates Auth0 JWT tokens."""

    def __init__(self):
        self.domain = settings.auth0_domain
        self.audience = settings.auth0_api_audience
        self.algorithms = settings.auth0_algorithms
        self._jwks = None
        self._jwks_url = f"https://{self.domain}/.well-known/jwks.json"

    def _get_jwks(self) -> Dict:
        """Fetch and cache JWKS from Auth0."""
        if not self._jwks:
            response = requests.get(self._jwks_url)
            response.raise_for_status()
            self._jwks = response.json()
        return self._jwks

    def _get_signing_key(self, token: str) -> str:
        """Extract signing key from token header."""
        try:
            unverified_header = jwt.get_unverified_header(token)
            kid = unverified_header["kid"]

            jwks = self._get_jwks()
            for key in jwks["keys"]:
                if key["kid"] == kid:
                    return jwt.algorithms.RSAAlgorithm.from_jwk(key)

            raise HTTPException(status_code=401, detail="Unable to find signing key")

        except Exception as e:
            logger.error("auth.signing_key_error", error=str(e))
            raise HTTPException(status_code=401, detail="Invalid token header")

    def validate_token(self, token: str) -> Dict[str, Any]:
        """
        Validate JWT token and return payload.

        Args:
            token: JWT token string

        Returns:
            Decoded token payload
        """
        try:
            signing_key = self._get_signing_key(token)

            payload = jwt.decode(
                token,
                signing_key,
                algorithms=self.algorithms,
                audience=self.audience,
                issuer=f"https://{self.domain}/",
            )

            logger.info("auth.token_validated", user_id=payload.get("sub"))
            return payload

        except InvalidTokenError as e:
            logger.warning("auth.token_invalid", error=str(e))
            raise HTTPException(status_code=401, detail="Invalid token")
        except Exception as e:
            logger.error("auth.validation_error", error=str(e))
            raise HTTPException(status_code=401, detail="Token validation failed")


# Global validator instance
auth_validator = Auth0Validator()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security),
) -> Dict[str, Any]:
    """
    Dependency to validate JWT and return user info.

    Usage:
        @app.get("/protected")
        async def protected_route(user: dict = Depends(get_current_user)):
            return {"user_id": user["sub"]}
    """
    token = credentials.credentials
    return auth_validator.validate_token(token)


async def require_scope(required_scope: str):
    """Factory for scope-based authorization."""

    async def scope_checker(user: Dict = Depends(get_current_user)) -> Dict:
        scopes = user.get("scope", "").split()
        if required_scope not in scopes:
            logger.warning(
                "auth.scope_denied",
                user_id=user.get("sub"),
                required=required_scope,
                available=scopes,
            )
            raise HTTPException(
                status_code=403, detail=f"Missing required scope: {required_scope}"
            )
        return user

    return scope_checker

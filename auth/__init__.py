"""
auth/
=====
Authentication and authorisation for the ACD Framework API.

Modules
-------
    jwt_handler.py   Sign and verify JWT tokens (HS256)
    password.py      bcrypt password hashing and verification
    models.py        User, Role, Session SQLAlchemy models
    router.py        POST /auth/login, /refresh, /logout
    guards.py        role_required, admin_only FastAPI dependencies
"""
from auth.jwt_handler import sign_token, verify_token, create_refresh_token
from auth.password import hash_password, verify_password
from auth.guards import role_required, admin_only

__all__ = [
    "sign_token", "verify_token", "create_refresh_token",
    "hash_password", "verify_password",
    "role_required", "admin_only",
]
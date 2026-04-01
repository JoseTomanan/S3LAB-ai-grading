import os

# Must be set before any app imports so core.database reads it when first imported
os.environ["DATABASE_URL"] = "sqlite://"

from sqlmodel import create_engine
from sqlalchemy.pool import StaticPool

import core.database as _db

# Replace the module-level engine with an in-memory one that uses a single
# shared connection (StaticPool) — required so that the test fixture's Session
# and the app's get_session() both see the same in-memory database.
_db.engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)

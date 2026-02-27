import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cognee.infrastructure.databases.relational import get_relational_engine
from sqlalchemy import text
from typing import cast


def main():
    db = get_relational_engine()
    print("Got Relational Engine.")

    session = cast(Any, db).engine.connect() if hasattr(db, "engine") else None

    if session:
        with session.begin():
            # Check edge tables
            try:
                res = session.execute(text("SELECT COUNT(*) FROM edges")).scalar()
                print("Total edges in DB:", res)
            except Exception as e:
                print("Failed to count edges:", e)


if __name__ == "__main__":
    main()

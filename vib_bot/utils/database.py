# utils/database.py

import sqlite3
import logging

# Configure module-level logger
db_logger = logging.getLogger("utils.database")
db_logger.setLevel(logging.INFO)
if not db_logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    db_logger.addHandler(ch)


def get_connection(db_path: str) -> sqlite3.Connection:
    """
    Returns a SQLite Connection with WAL journal mode and busy timeout.
    """
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=30000;")
    return conn


def ensure_master_schema(db_path: str) -> None:
    """
    Ensures the master database has the necessary tables for predictions,
    pending_feedback, and feedback, and adds a 'symbol' column to each if missing.
    """
    with get_connection(db_path) as conn:
        cur = conn.cursor()

        # Create tables if they don't exist, including the symbol column
        cur.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                symbol TEXT,
                predicted_label INTEGER,
                model_id TEXT
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS pending_feedback (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                symbol TEXT,
                predicted_label INTEGER,
                features TEXT,
                model_id TEXT
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                symbol TEXT,
                predicted_label INTEGER,
                true_label INTEGER
            );
        """)
        conn.commit()

        # Ensure 'symbol' column exists on each table (for older installs)
        for table in ("predictions", "pending_feedback", "feedback"):
            cur.execute(f"PRAGMA table_info({table});")
            existing_cols = [row[1] for row in cur.fetchall()]
            if "symbol" not in existing_cols:
                db_logger.info(f"Adding 'symbol' column to {table}")
                cur.execute(f"ALTER TABLE {table} ADD COLUMN symbol TEXT;")
        conn.commit()

    db_logger.info(f"Master schema ensured in {db_path}")


def migrate(db_path: str, migrations: dict[int, str]) -> None:
    """
    Applies incremental migrations stored in a dict mapping version->SQL.
    Maintains a _migrations table to track applied versions.
    """
    with get_connection(db_path) as conn:
        cur = conn.cursor()
        # ensure migrations table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS _migrations (
                version INTEGER PRIMARY KEY,
                applied_at TEXT
            );
        """)
        cur.execute("SELECT version FROM _migrations;")
        applied = {row[0] for row in cur.fetchall()}

        for version, stmt in sorted(migrations.items()):
            if version in applied:
                continue
            db_logger.info(f"Applying migration {version}")
            cur.executescript(stmt)
            cur.execute(
                "INSERT INTO _migrations(version, applied_at) VALUES (?, CURRENT_TIMESTAMP);",
                (version,)
            )
        conn.commit()

    db_logger.info(f"Migrations up to {max(migrations.keys(), default=0)} applied on {db_path}")
"""
scripts/seed_db.py
==================
Seed the ACD Framework database with initial data.

Creates:
    - Default admin user (admin / admin123)
    - Default analyst user (analyst / analyst123)
    - Sample incidents for dashboard demo
    - Sample training run record

Usage
-----
    python scripts/seed_db.py

    # Custom admin password
    python scripts/seed_db.py --admin-password my_secure_password

    # Seed only users (skip sample data)
    python scripts/seed_db.py --users-only

    # Reset: drop all data and re-seed
    python scripts/seed_db.py --reset
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from monitoring.structlog_config import configure_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Seed ACD Framework database.")
    p.add_argument("--admin-password",   default="admin123")
    p.add_argument("--analyst-password", default="analyst123")
    p.add_argument("--users-only",       action="store_true")
    p.add_argument("--reset",            action="store_true", help="Drop all tables first")
    p.add_argument("--verbose",          action="store_true")
    return p.parse_args()


async def seed_users(admin_pw: str, analyst_pw: str) -> None:
    """Create default admin and analyst users."""
    from db.session import get_db_session
    from db.models.user import User
    from auth.password import hash_password
    from sqlalchemy import select

    session = await get_db_session()
    async with session:
        for username, email, password, role in [
            ("admin",   "admin@acd.local",   admin_pw,   "admin"),
            ("analyst", "analyst@acd.local", analyst_pw, "analyst"),
            ("viewer",  "viewer@acd.local",  "viewer123","viewer"),
        ]:
            # Check if user already exists
            result = await session.execute(
                select(User).where(User.username == username)
            )
            existing = result.scalar_one_or_none()

            if existing:
                logger.info("User %r already exists — skipping.", username)
                continue

            user = User(
                username      = username,
                email         = email,
                password_hash = hash_password(password),
                role          = role,
                is_active     = True,
                created_at    = time.time(),
            )
            session.add(user)
            logger.info("Created user: %r (role=%s)", username, role)

        await session.commit()


async def seed_sample_incidents() -> None:
    """Create sample incident records for dashboard demo."""
    from db.session import get_db_session
    from db.models.incident import Incident

    sample_incidents = [
        {
            "incident_id":   "INC-20240115-0001",
            "title":         "Log4Shell Exploitation — Host-3",
            "severity":      "CRITICAL",
            "report_type":   "INCIDENT",
            "cve_ids":       "CVE-2021-44228",
            "technique_ids": "T1190,T1059",
            "hosts_affected":"Host-3,Host-1",
            "markdown":      "# Log4Shell Exploitation\n\nCritical incident detected...",
        },
        {
            "incident_id":   "INC-20240116-0001",
            "title":         "Concept Drift — Attacker Pattern Shift",
            "severity":      "HIGH",
            "report_type":   "DRIFT",
            "cve_ids":       "",
            "technique_ids": "T1046,T1021",
            "hosts_affected":"",
            "markdown":      "# Concept Drift Event\n\nDistribution shift detected...",
        },
        {
            "incident_id":   "INC-20240117-0001",
            "title":         "APT Lateral Movement — Enterprise Subnet",
            "severity":      "HIGH",
            "report_type":   "INCIDENT",
            "cve_ids":       "CVE-2021-26855",
            "technique_ids": "T1021,T1543",
            "hosts_affected":"Enterprise0,User2",
            "markdown":      "# APT Lateral Movement\n\nTargeted APT detected...",
        },
    ]

    session = await get_db_session()
    async with session:
        for data in sample_incidents:
            incident = Incident(
                **data,
                generated_at = time.time() - (24 * 3600),  # yesterday
            )
            session.add(incident)
        await session.commit()

    logger.info("Seeded %d sample incidents.", len(sample_incidents))


async def seed_training_run() -> None:
    """Create a sample training run record."""
    from db.session import get_db_session
    from db.models.training_run import TrainingRun

    session = await get_db_session()
    async with session:
        run = TrainingRun(
            run_id            = str(uuid.uuid4())[:8],
            agent_type        = "cvar_ppo",
            total_timesteps   = 2_000_000,
            final_mean_reward = 8.74,
            final_cvar_005    = -2.14,
            drift_events      = 2,
            ewc_tasks         = 3,
            elapsed_s         = 7200.0,
            started_at        = time.time() - 9000,
            completed_at      = time.time() - 1800,
            is_running        = False,
            config_json       = json.dumps({"agent_type": "cvar_ppo", "alpha": 0.05}),
        )
        session.add(run)
        await session.commit()
    logger.info("Seeded sample training run.")


async def init_and_seed(args: argparse.Namespace) -> None:
    """Main async seeding function."""
    from db.session import init_db

    # Initialise DB tables
    logger.info("Initialising database tables...")
    await init_db()

    # Seed users
    logger.info("Seeding users...")
    await seed_users(args.admin_password, args.analyst_password)

    if not args.users_only:
        logger.info("Seeding sample incidents...")
        try:
            await seed_sample_incidents()
        except Exception as exc:
            logger.warning("Could not seed incidents: %s", exc)

        logger.info("Seeding sample training run...")
        try:
            await seed_training_run()
        except Exception as exc:
            logger.warning("Could not seed training run: %s", exc)


def main() -> None:
    args = parse_args()
    configure_logging(level="DEBUG" if args.verbose else "INFO")

    logger.info("Seeding ACD Framework database...")
    asyncio.run(init_and_seed(args))

    print("\n" + "="*55)
    print("DATABASE SEEDED")
    print("="*55)
    print("  Users created:")
    print("    admin    / " + args.admin_password + "  (role: admin)")
    print("    analyst  / " + args.analyst_password + "  (role: analyst)")
    print("    viewer   / viewer123  (role: viewer)")
    if not args.users_only:
        print("  Sample incidents: 3")
        print("  Sample training runs: 1")
    print("\n  Login: POST /auth/login")
    print("    {\"username\": \"admin\", \"password\": \"" + args.admin_password + "\"}")
    print("="*55 + "\n")


if __name__ == "__main__":
    main()
"""One-shot script: close all open short positions and clean up local DB records.

Run once from the project root:
    python scripts/close_all_shorts.py

Dry-run (prints plan without placing orders):
    python scripts/close_all_shorts.py --dry-run
"""

import logging
import sys

sys.path.insert(0, ".")

import argparse

from execution import trader

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
logger = logging.getLogger(__name__)


def main(dry_run: bool) -> None:
    client = trader.get_client()

    positions = client.get_all_positions()
    shorts = [p for p in positions if float(p.qty) < 0]

    if not shorts:
        logger.info("No short positions found — nothing to do.")
        return

    logger.info(f"Found {len(shorts)} short position(s) to close:")
    for p in shorts:
        qty = abs(float(p.qty))
        pnl = float(p.unrealized_pl) if p.unrealized_pl else 0.0
        logger.info(f"  {p.symbol:8s}  qty={qty:.0f}  P&L=${pnl:+.2f}")

    if dry_run:
        logger.info("Dry run — no orders placed.")
        return

    total_pnl = 0.0
    closed, failed = [], []

    for p in shorts:
        sym = p.symbol
        pnl = float(p.unrealized_pl) if p.unrealized_pl else 0.0
        logger.info(f"Closing {sym} ...")
        result = trader.close_position(client, sym)
        if result.is_success:
            trader.record_cover(sym)
            total_pnl += pnl
            closed.append(sym)
            logger.info(f"  {sym} closed  P&L=${pnl:+.2f}")
        else:
            failed.append(sym)
            logger.error(f"  {sym} FAILED: {result.rejection_reason or result.status}")

    logger.info(f"\nDone. Closed {len(closed)}/{len(shorts)}. Realised P&L: ${total_pnl:+.2f}")
    if failed:
        logger.error(f"Failed to close: {failed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(dry_run=args.dry_run)

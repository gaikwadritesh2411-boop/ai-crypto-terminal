"""Headless paper-trading worker: keeps trading while no browser is open.

    python worker.py            # loop forever, one cycle every 60 s
    python worker.py --once     # single cycle (for cron / smoke tests)
"""
from __future__ import annotations

import argparse
import logging
import time

from trader.config import Config
from trader.data import MarketData
from trader.engine import TradingEngine
from trader.store import Store

log = logging.getLogger("worker")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--once", action="store_true")
    p.add_argument("--interval", type=int, default=60)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = Config()
    engine = TradingEngine(cfg, Store(cfg.db_path), MarketData(cfg.exchange_id))
    log.info("PAPER TRADING worker started (exchange=%s, db=%s)", cfg.exchange_id, cfg.db_path)
    while True:
        try:
            r = engine.run_cycle()
            log.info("cycle ok: equity=%.2f opened=%d closed=%d errors=%d", r.equity, len(r.opened),
                     len(r.closed), len(r.errors))
            for e in r.errors:
                log.warning(e)
        except Exception:
            log.exception("cycle failed; will retry")
        if a.once:
            return 0
        time.sleep(a.interval)


if __name__ == "__main__":
    raise SystemExit(main())

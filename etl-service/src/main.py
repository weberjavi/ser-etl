"""ETL Service entry point."""

import time
import signal
import sys
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
import structlog

from .config import Config
from .sync import SyncManager

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


class ETLLauncher:
    """Manages the ETL service lifecycle."""

    def __init__(self):
        self.config = Config.from_env()
        self.sync_manager: SyncManager = None
        self.scheduler: BackgroundScheduler = None
        self._shutdown = False

    def setup_signal_handlers(self):
        """Setup graceful shutdown handlers."""

        def signal_handler(signum, frame):
            logger.info("shutdown.signal_received", signal=signum)
            self._shutdown = True
            if self.scheduler:
                self.scheduler.shutdown(wait=True)
            if self.sync_manager:
                self.sync_manager.close()
            sys.exit(0)

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

    def run_sync_job(self):
        """Wrapper for sync job with error handling."""
        try:
            success = self.sync_manager.run_sync()
            if not success:
                logger.warning("sync_job.failed")
        except Exception as e:
            logger.error("sync_job.exception", error=str(e), exc_info=True)

    def start(self):
        """Start the ETL service."""
        logger.info(
            "etl.starting",
            interval_seconds=self.config.sync_interval_seconds,
            tables=self.config.tables_to_sync,
        )

        # Initialize sync manager
        self.sync_manager = SyncManager(self.config)
        self.sync_manager.initialize()

        # Setup scheduler
        self.scheduler = BackgroundScheduler()
        self.scheduler.add_job(
            self.run_sync_job,
            trigger=IntervalTrigger(seconds=self.config.sync_interval_seconds),
            id="sync_job",
            replace_existing=True,
            max_instances=1,  # Prevent overlapping runs
            coalesce=True,  # Skip missed runs if behind
        )

        # Run initial sync immediately
        self.run_sync_job()

        # Start scheduler
        self.scheduler.start()

        logger.info("etl.started")

        # Keep main thread alive
        try:
            while not self._shutdown:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            self.scheduler.shutdown()
            self.sync_manager.close()


def main():
    """Entry point."""
    launcher = ETLLauncher()
    launcher.setup_signal_handlers()
    launcher.start()


if __name__ == "__main__":
    main()

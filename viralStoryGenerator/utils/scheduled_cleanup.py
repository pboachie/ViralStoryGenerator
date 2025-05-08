# viralStoryGenerator/utils/scheduled_cleanup.py
"""Scheduled task runner for periodic file cleanup."""

import time
from typing import Any, Dict, Optional
import schedule
import threading # Still used if running schedule in background thread
import os
from datetime import datetime, timezone # Use timezone

from viralStoryGenerator.src.logger import logger as _logger
from viralStoryGenerator.utils.config import config as appconfig
# Import storage manager instance directly
from viralStoryGenerator.utils.storage_manager import storage_manager

class ScheduledCleanupTask:
    """Runs storage_manager.cleanup_old_files periodically."""

    def __init__(self):
        """Initializes the cleanup task state."""
        self.is_running = False
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.interval_hours = appconfig.storage.CLEANUP_INTERVAL_HOURS
        self.retention_days = appconfig.storage.FILE_RETENTION_DAYS
        self.retention_enabled = isinstance(self.retention_days, int) and self.retention_days > 0

        # Metrics (optional)
        self.last_run_time: Optional[datetime] = None
        self.last_deleted_count: int = 0
        self.total_deleted_count: int = 0

    def start(self):
        """Starts the scheduler in a background thread if not already running."""
        if not self.retention_enabled:
            _logger.info("File retention cleanup is disabled (retention_days <= 0). Task not started.")
            return False

        if self.is_running:
            _logger.warning("Scheduled cleanup task is already running.")
            return False

        _logger.info(f"Starting scheduled cleanup task. Interval: {self.interval_hours} hours, Retention: {self.retention_days} days.")
        self.is_running = True
        self._stop_event.clear()

        # --- Run scheduler in a separate thread ---
        # This is required because schedule.run_pending() blocks within its loop.
        # Cannot easily integrate schedule library with FastAPI's asyncio loop directly.
        self._thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        """Signals the background scheduler thread to stop."""
        if not self.is_running: return
        _logger.info("Stopping scheduled cleanup task...")
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)
            if self._thread.is_alive():
                 _logger.warning("Cleanup scheduler thread did not exit gracefully.")
        self.is_running = False
        _logger.info("Scheduled cleanup task stopped.")

    def _scheduler_loop(self):
        """Background thread method that runs the schedule loop."""
        _logger.info("Cleanup scheduler thread started.")
        # Run once immediately on start
        self._run_cleanup_job()

        # Schedule the job
        try:
            schedule.every(self.interval_hours).hours.do(self._run_cleanup_job)
        except Exception as e:
             _logger.exception(f"Failed to schedule cleanup job: {e}. Thread exiting.")
             self.is_running = False
             return

        # Run pending jobs periodically until stop signal
        while not self._stop_event.is_set():
            try:
                schedule.run_pending()
            except Exception as e:
                 # Catch errors within run_pending itself if any
                 _logger.exception(f"Error during schedule.run_pending(): {e}")
            wait_time = min(schedule.idle_seconds() if schedule.get_jobs() else 60, 5.0)
            self._stop_event.wait(timeout=wait_time)

        _logger.info("Cleanup scheduler thread finished.")


    def _run_cleanup_job(self):
        """The actual cleanup job executed by the scheduler."""
        _logger.info("Executing scheduled file cleanup job...")
        start_time = time.time()
        deleted_count = 0
        try:
            deleted_count = storage_manager.cleanup_old_files(self.retention_days)

            # Update metrics
            self.last_run_time = datetime.now(timezone.utc)
            self.last_deleted_count = deleted_count
            self.total_deleted_count += deleted_count

            duration = time.time() - start_time
            _logger.info(f"Scheduled cleanup job completed in {duration:.2f}s. Deleted {deleted_count} files/objects.")
            return True
        except Exception as e:
            duration = time.time() - start_time
            _logger.exception(f"Error during scheduled cleanup job execution (duration: {duration:.2f}s): {e}")
            return False

    def get_status(self) -> Dict[str, Any]:
        """Returns the current status and metrics of the cleanup task."""
        return {
            "is_running": self.is_running,
            "retention_enabled": self.retention_enabled,
            "retention_days": self.retention_days,
            "interval_hours": self.interval_hours,
            "last_run_time_utc": self.last_run_time.isoformat() if self.last_run_time else None,
            "last_deleted_count": self.last_deleted_count,
            "total_deleted_count": self.total_deleted_count,
            "next_run_time_utc": schedule.next_run.isoformat() if self.is_running and schedule.next_run else None
        }


cleanup_task = ScheduledCleanupTask()
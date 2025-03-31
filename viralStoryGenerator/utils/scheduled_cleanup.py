#!/usr/bin/env python
# viralStoryGenerator/utils/scheduled_cleanup.py

import time
import schedule
import threading
import os
from datetime import datetime

from viralStoryGenerator.src.logger import logger as _logger
from viralStoryGenerator.utils.config import config
from viralStoryGenerator.utils.storage_manager import storage_manager

class ScheduledCleanupTask:
    """Scheduled task to clean up old files based on retention policy"""

    def __init__(self):
        """Initialize the scheduled cleanup task"""
        self.is_running = False
        self.cleanup_thread = None
        self.interval_hours = config.storage.CLEANUP_INTERVAL_HOURS
        self.retention_days = config.storage.FILE_RETENTION_DAYS
        self.retention_enabled = self.retention_days > 0

        # Set up metrics for monitoring cleanups
        self.last_run_time = None
        self.last_deleted_count = 0
        self.total_deleted_count = 0

    def start(self):
        """Start the scheduled cleanup task"""
        if not self.retention_enabled:
            _logger.info("File retention is disabled (retention days is 0). Scheduled cleanup not started.")
            return False

        if self.is_running:
            _logger.warning("Scheduled cleanup task is already running")
            return False

        self.is_running = True

        # Schedule the cleanup at the specified interval
        schedule.every(self.interval_hours).hours.do(self._run_cleanup)

        # Start the scheduler in a background thread
        self.cleanup_thread = threading.Thread(target=self._scheduler_thread, daemon=True)
        self.cleanup_thread.start()

        _logger.info(f"Scheduled cleanup task started. Interval: {self.interval_hours} hours, Retention: {self.retention_days} days")
        return True

    def stop(self):
        """Stop the scheduled cleanup task"""
        if not self.is_running:
            return

        self.is_running = False
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            # Let the thread terminate naturally on next iteration
            self.cleanup_thread.join(timeout=10)

        _logger.info("Scheduled cleanup task stopped")

    def _scheduler_thread(self):
        """Background thread for running scheduled tasks"""
        _logger.info("Cleanup scheduler thread started")

        # Run cleanup once on startup
        self._run_cleanup()

        while self.is_running:
            schedule.run_pending()
            time.sleep(60)  # Check for scheduled tasks every minute

    def _run_cleanup(self):
        """Run the actual cleanup task"""
        try:
            _logger.info("Running scheduled file cleanup")
            deleted_count = storage_manager.cleanup_old_files(self.retention_days)

            # Update metrics
            self.last_run_time = datetime.now()
            self.last_deleted_count = deleted_count
            self.total_deleted_count += deleted_count

            _logger.info(f"Scheduled cleanup completed. Deleted {deleted_count} files.")
            return True
        except Exception as e:
            _logger.error(f"Error during scheduled cleanup: {e}")
            return False

    def status(self):
        """Get the status of the cleanup task"""
        return {
            "is_running": self.is_running,
            "retention_days": self.retention_days,
            "interval_hours": self.interval_hours,
            "last_run": self.last_run_time.isoformat() if self.last_run_time else None,
            "last_deleted_count": self.last_deleted_count,
            "total_deleted_count": self.total_deleted_count
        }


# Create a single instance of the cleanup task
cleanup_task = ScheduledCleanupTask()
#!/usr/bin/env python
"""
Standalone script to start the Crawl4AI Redis Queue Worker.
This script can be used to start the worker process separately from the main application.

Usage:
  python start_queue_worker.py
"""
import sys
import os

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.queue_worker import main

if __name__ == "__main__":
    main()
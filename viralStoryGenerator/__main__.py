#!/usr/bin/env python
import sys
from viralStoryGenerator.main import main
import logging

import viralStoryGenerator.src.logger
_logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Check for subcommands (api, worker, etc.)
    if len(sys.argv) > 1 and sys.argv[1] == "api":
        # Remove the 'api' argument so main doesn't receive it
        sys.argv.pop(1)
        _logger.info("Starting API server...")

    main()

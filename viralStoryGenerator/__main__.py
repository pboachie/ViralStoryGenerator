#!/usr/bin/env python
import sys
from viralStoryGenerator.main import main
from viralStoryGenerator.src.logger import logger as _logger

if __name__ == "__main__":
    # Check for subcommands (api, worker, etc.)
    if len(sys.argv) > 1 and sys.argv[1] == "api":
        # Remove the 'api' argument so main doesn't receive it
        sys.argv.pop(1)
        _logger.info("Starting API server...")

    # Run the main function
    main()

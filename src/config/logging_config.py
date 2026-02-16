"""
Logging Configuration

Configures logging for CloudWatch and console output.
"""

import logging
import sys


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """
    Configure and return the application logger.

    Args:
        level: Logging level (default: logging.INFO)

    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )
    return logging.getLogger(__name__)


logger = setup_logging()

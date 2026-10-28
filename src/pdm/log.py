"""Structured logging setup: one key=value line per event on stderr.

Call get_logger(__name__) in each module; configure() sets the format once.
"""

import logging
import sys

_FORMAT = "%(asctime)s level=%(levelname)s logger=%(name)s %(message)s"


def configure(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_FORMAT))
        root.addHandler(handler)
    root.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

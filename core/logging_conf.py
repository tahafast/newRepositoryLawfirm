"""
Logging Configuration
Centralized logging setup for the Law Firm Chatbot
"""

import logging
import os

level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, level, logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

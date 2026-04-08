import sys
import os

# Ensure the server directory is in sys.path for internal imports
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SERVER_PATH = os.path.join(PROJECT_ROOT, "server")

if SERVER_PATH not in sys.path:
    sys.path.insert(0, SERVER_PATH)

# Import the registry from the server module
try:
    from graders import tasks_with_graders
except ImportError:
    # Fallback if graders is not directly importable
    sys.path.append(SERVER_PATH)
    from server.graders import tasks_with_graders

# Export the list for platform discovery
__all__ = ["tasks_with_graders"]

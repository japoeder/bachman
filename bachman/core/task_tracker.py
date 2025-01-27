"""Task tracking implementation for async operations."""

import logging
from typing import Dict, Optional
from datetime import datetime
from bachman.core.interfaces import TaskTracker

logger = logging.getLogger(__name__)


class AsyncTaskTracker(TaskTracker):
    """Tracks status of async processing tasks."""

    def __init__(self):
        self.tasks = {}

    async def update_status(self, task_id: str, status: Dict) -> None:
        """Update the status of a task."""
        try:
            status["last_updated"] = datetime.utcnow().isoformat()
            self.tasks[task_id] = status
            logger.debug(f"Updated status for task {task_id}: {status}")
        except Exception as e:
            logger.error(f"Error updating task status: {str(e)}")
            raise

    async def get_status(self, task_id: str) -> Optional[Dict]:
        """Get the current status of a task."""
        try:
            return self.tasks.get(task_id)
        except Exception as e:
            logger.error(f"Error getting task status: {str(e)}")
            raise

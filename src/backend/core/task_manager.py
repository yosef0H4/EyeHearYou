"""Task management for cancellation and status updates"""
import threading
import time
from queue import Queue
from typing import Optional


class TaskManager:
    def __init__(self):
        self._cancel_flag = threading.Event()
        self._is_running = False
        self._current_task_name = ""
        # Queue for SSE messages (status updates)
        self.message_queue = Queue()

    def start_task(self, name: str):
        """Start a new task, cancelling any previous one"""
        if self._is_running:
            self.cancel_current_task()
            
        self._cancel_flag.clear()
        self._is_running = True
        self._current_task_name = name
        self.emit_status(f"Starting {name}...", progress=0)
        print(f"[TaskManager] Starting task: {name}")

    def cancel_current_task(self):
        """Signal the current task to stop"""
        if self._is_running:
            self._cancel_flag.set()
            self.emit_status("Cancelling...", is_loading=False)
            print(f"[TaskManager] Cancellation requested for: {self._current_task_name}")
            # Give it a moment to react
            time.sleep(0.1)

    def finish_task(self):
        """Mark current task as complete"""
        self._is_running = False
        self.emit_status("Ready", is_loading=False, progress=100)

    def is_cancelled(self) -> bool:
        """Check if the current task should stop"""
        return self._cancel_flag.is_set()

    def is_running(self) -> bool:
        """Check if a task is currently running"""
        return self._is_running

    def emit_status(self, message: str, progress: Optional[float] = None, is_loading: bool = True):
        """Send a status update to the frontend via SSE"""
        data = {
            "type": "status",
            "message": message,
            "isLoading": is_loading
        }
        if progress is not None:
            data["progress"] = progress
            
        self.message_queue.put(data)


# Global instance
task_manager = TaskManager()






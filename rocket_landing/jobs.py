"""Run a single owned worker process without blocking the control panel."""

import os
import queue
import subprocess
import threading
from pathlib import Path


class JobRunner:
    def __init__(self, root, python):
        self.root = Path(root)
        self.python = Path(python)
        if self.python.name.lower() == "pythonw.exe":
            self.python = self.python.with_name("python.exe")
        self.events = queue.Queue()
        self.process = None
        self.cancelled = False

    def start(self, arguments):
        if self.process is not None:
            raise RuntimeError("Finish or stop the current task before starting another")
        environment = dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1")
        process = subprocess.Popen(
            [str(self.python), "-u", "-m", "rocket_landing", *map(str, arguments)],
            cwd=self.root, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", shell=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.process = process
        self.cancelled = False

        def receive():
            try:
                for line in process.stdout:
                    self.events.put(("line", process.pid, line))
            finally:
                process.stdout.close()
                code = process.wait()
                self.events.put(("done", process.pid, code))

        threading.Thread(target=receive, daemon=True).start()

    def cancel(self):
        if self.process is not None and self.process.poll() is None:
            self.cancelled = True
            self.process.terminate()

    def acknowledge(self, pid):
        if self.process is not None and self.process.pid == pid:
            self.process = None

    def close(self):
        self.cancel()
        if self.process is not None:
            try:
                self.process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self.process.kill()


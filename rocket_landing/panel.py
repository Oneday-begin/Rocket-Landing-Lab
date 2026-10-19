"""A local desktop control panel; no web account or server required."""

import os
import queue
import sys
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk

from .catalog import discover_models
from .jobs import JobRunner
from .scenarios import SCENARIOS


class ControlPanel:
    def __init__(self, window, root=None):
        self.window = window
        self.root = Path(root or Path(__file__).resolve().parent.parent)
        self.runner = JobRunner(self.root, sys.executable)
        self.models = []
        self.task_buttons = []
        self.last_report = None
        self.last_recording = None
        self.pending_output = None
        self.pending_kind = None
        self.weather = tk.StringVar(value="calm")
        self.episodes = tk.StringVar(value="20")
        self.seed = tk.StringVar(value="200000")
        self.status = tk.StringVar(value="Ready — choose a model below")
        window.title("Rocket Landing Lab • Control Panel")
        height = min(820, max(680, window.winfo_screenheight() - 100))
        window.geometry(f"1080x{height}")
        window.minsize(940, 680)
        window.configure(background="#0b1120")
        self._theme()
        self._build()
        self.refresh()
        window.protocol("WM_DELETE_WINDOW", self.close)
        window.after(100, self._poll)

    def _theme(self):
        style = ttk.Style(self.window)
        style.theme_use("clam")
        style.configure("TFrame", background="#0b1120")
        style.configure("TLabel", background="#0b1120", foreground="#e2e8f0",
                        font=("Segoe UI", 11))
        style.configure("Title.TLabel", font=("Segoe UI", 24, "bold"), foreground="#67e8f9")
        style.configure("Muted.TLabel", foreground="#94a3b8", font=("Segoe UI", 10))
        style.configure("TButton", padding=(12, 9), font=("Segoe UI", 10))
        style.configure("Treeview", background="#152034", fieldbackground="#152034",
                        foreground="#e2e8f0", rowheight=29, font=("Segoe UI", 10))
        style.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"), padding=8)
        style.map("Treeview", background=[("selected", "#155e75")],
                  foreground=[("selected", "#ffffff")])

    def _button(self, parent, label, callback, task=False):
        button = ttk.Button(parent, text=label, command=callback)
        button.pack(side="left", padx=(0, 8), pady=3)
        if task:
            self.task_buttons.append(button)
        return button

    def _build(self):
        shell = ttk.Frame(self.window)
        shell.pack(fill="both", expand=True)
        canvas = tk.Canvas(shell, background="#0b1120", highlightthickness=0)
        page_scroll = ttk.Scrollbar(shell, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=page_scroll.set)
        page_scroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        frame = ttk.Frame(canvas, padding=20)
        content = canvas.create_window(0, 0, window=frame, anchor="nw")
        frame.bind("<Configure>", lambda event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(
            content, width=event.width, height=max(frame.winfo_reqheight(), event.height)))
        ttk.Label(frame, text="Rocket Landing Lab", style="Title.TLabel").pack(anchor="w")
        ttk.Label(frame, text="Watch trained rockets. Test them in changing weather. Compare measured results.",
                  style="Muted.TLabel").pack(anchor="w", pady=(2, 16))
        ttk.Label(frame, text="1. Choose a saved model").pack(anchor="w")
        table_frame = ttk.Frame(frame)
        table_frame.pack(fill="x", pady=(8, 4))
        self.table = ttk.Treeview(table_frame, columns=("model", "score", "landed", "episodes"),
                                  show="headings", height=5, selectmode="browse")
        for column, title, width in (("model", "Model / training seed / checkpoint", 470),
                                      ("score", "Original score", 140),
                                      ("landed", "Landed", 130), ("episodes", "Episodes", 110)):
            self.table.heading(column, text=title)
            self.table.column(column, width=width, anchor="w" if column == "model" else "center")
        table_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=table_scroll.set)
        table_scroll.pack(side="right", fill="y")
        self.table.pack(side="left", fill="x", expand=True)
        ttk.Label(frame, text="Scores above are saved calm-weather evaluations. An untested checkpoint shows “Not tested”.",
                  style="Muted.TLabel").pack(anchor="w")
        ttk.Label(frame, text="2. Set your test", padding=(0, 14, 0, 6)).pack(anchor="w")
        settings = ttk.Frame(frame)
        settings.pack(fill="x")
        ttk.Label(settings, text="Weather").pack(side="left", padx=(0, 8))
        ttk.Combobox(settings, textvariable=self.weather, values=list(SCENARIOS),
                     state="readonly", width=12).pack(side="left", padx=(0, 20))
        ttk.Label(settings, text="Test episodes").pack(side="left", padx=(0, 8))
        ttk.Spinbox(settings, textvariable=self.episodes, from_=1, to=1000,
                     width=7).pack(side="left", padx=(0, 20))
        ttk.Label(settings, text="First episode seed").pack(side="left", padx=(0, 8))
        ttk.Entry(settings, textvariable=self.seed, width=12).pack(side="left")
        ttk.Label(frame, text="Wind tests run every condition: calm → breezy → gusty. Watch and record play up to 3 episodes.",
                  style="Muted.TLabel").pack(anchor="w", pady=(8, 12))
        actions = ttk.Frame(frame)
        actions.pack(fill="x")
        self._button(actions, "▶ Watch AI", self.watch, task=True)
        self._button(actions, "Record GIF", self.record, task=True)
        self._button(actions, "Evaluate selected", self.evaluate, task=True)
        self._button(actions, "Wind-test selected", self.wind_selected, task=True)
        self._button(actions, "Compare finals in wind", self.wind_all, task=True)
        self.stop_button = self._button(actions, "Stop", self.stop)
        self.stop_button.state(["disabled"])
        quality_actions = ttk.Frame(frame)
        quality_actions.pack(fill="x")
        self._button(quality_actions, "Run quality gate (all finals)", self.quality, task=True)
        ttk.Label(quality_actions, text="At least 50 flights per weather condition; checks every final model.",
                  style="Muted.TLabel").pack(side="left")
        links = ttk.Frame(frame)
        links.pack(fill="x", pady=(8, 12))
        self._button(links, "Original results", lambda: self.open_file(self.root / "artifacts/report.html"))
        self._button(links, "Latest test report", self.open_latest_report)
        self._button(links, "Recordings", self.open_recordings)
        self._button(links, "Refresh models", self.refresh)
        self._button(links, "Project folder", lambda: self.open_file(self.root))
        ttk.Label(frame, textvariable=self.status, foreground="#67e8f9").pack(anchor="w", pady=(0, 6))
        self.progress = ttk.Progressbar(frame, mode="indeterminate")
        self.progress.pack(fill="x", pady=(0, 10))
        log_frame = ttk.Frame(frame)
        log_frame.pack(fill="both", expand=True)
        self.log = tk.Text(log_frame, background="#101a2c", foreground="#e2e8f0",
                           insertbackground="white", font=("Consolas", 10), height=5,
                           state="disabled", wrap="word", relief="flat", padx=12, pady=10)
        scrollbar = ttk.Scrollbar(log_frame, command=self.log.yview)
        self.log.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.log.pack(side="left", fill="both", expand=True)
        ttk.Label(frame, text="Tests save new results for each run. Closing this panel stops its running task.",
                  style="Muted.TLabel").pack(anchor="w", pady=(8, 0))

    def refresh(self):
        selected = self.selected(optional=True)
        selected_path = selected.path if selected else None
        self.models = discover_models(self.root)
        self.table.delete(*self.table.get_children())
        choice = "0"
        for index, model in enumerate(self.models):
            summary = model.summary
            self.table.insert("", "end", iid=str(index), values=(model.label,
                f"{summary['mean_base_return']:.1f}" if summary else "Not tested",
                f"{summary['landed_rate']:.0%}" if summary else "—",
                summary["episodes"] if summary else "—"))
            if model.path == selected_path:
                choice = str(index)
        if self.models:
            self.table.selection_set(choice)
        else:
            self.status.set("No saved models found. Run the training workflow first.")

    def selected(self, optional=False):
        selection = self.table.selection()
        if selection and int(selection[0]) < len(self.models):
            return self.models[int(selection[0])]
        if optional:
            return None
        raise ValueError("Select a saved model first.")

    def settings(self):
        try:
            episodes, seed = int(self.episodes.get()), int(self.seed.get())
        except ValueError:
            raise ValueError("Episodes and seed must be whole numbers.") from None
        if not 1 <= episodes <= 1000:
            raise ValueError("Choose between 1 and 1,000 episodes.")
        if not 0 <= seed <= 2**32 - 1001:
            raise ValueError("Choose a non-negative seed below 4,294,966,295.")
        return episodes, seed

    def _start(self, title, arguments, kind=None, output=None):
        try:
            self.runner.start(arguments)
        except (OSError, RuntimeError) as error:
            messagebox.showerror("Could not start", str(error), parent=self.window)
            return
        self.pending_kind, self.pending_output = kind, output
        self.status.set(title)
        self._append(f"\n{title}\n")
        self.progress.start(12)
        for button in self.task_buttons:
            button.state(["disabled"])
        self.stop_button.state(["!disabled"])

    def _inputs(self):
        try:
            return self.selected(), *self.settings()
        except ValueError as error:
            messagebox.showerror("Check your settings", str(error), parent=self.window)
            return None

    def _output(self, category):
        return self.root / "artifacts" / "control-panel" / category / datetime.now().strftime("%Y%m%d-%H%M%S-%f")

    def watch(self):
        values = self._inputs()
        if values:
            model, episodes, seed = values
            self._start("Playing the selected model…", ["watch", "--model", model.path,
                "--episodes", min(3, episodes), "--seed", seed, "--scenario", self.weather.get(), "--human"])

    def record(self):
        values = self._inputs()
        if values:
            model, episodes, seed = values
            output = self._output("recordings")
            self._start("Recording real simulator frames…", ["watch", "--model", model.path,
                "--episodes", min(3, episodes), "--seed", seed, "--scenario", self.weather.get(),
                "--output", output], "recording", output)

    def evaluate(self):
        values = self._inputs()
        if values:
            model, episodes, seed = values
            output = self._output("evaluations")
            self._start("Testing the selected policy…", ["evaluate", "--model", model.path,
                "--episodes", episodes, "--seed", seed, "--scenario", self.weather.get(),
                "--output", output], "evaluation", output)

    def wind_selected(self):
        self._wind(False)

    def quality(self):
        values = self._inputs()
        if values:
            _, episodes, seed = values
            models = [item.path for item in self.models if item.checkpoint == "final"]
            output = self._output("quality")
            self._start("Checking model quality across all weather conditions…",
                        ["quality", "--models", *models, "--episodes", max(50, episodes),
                         "--seed", seed, "--output", output], "quality", output)

    def wind_all(self):
        self._wind(True)

    def _wind(self, all_models):
        values = self._inputs()
        if not values:
            return
        model, episodes, seed = values
        models = [item.path for item in self.models if item.checkpoint == "final"] if all_models else [model.path]
        output = self._output("wind-tests")
        self._start(f"Wind test: {len(models)} model(s) × 3 weather conditions × {episodes} episodes…",
            ["robustness", "--models", *models, "--episodes", episodes, "--seed", seed,
             "--output", output], "wind", output)

    def stop(self):
        self.runner.cancel()
        self.status.set("Stopping the current task…")

    def _append(self, message):
        self.log.configure(state="normal")
        self.log.insert("end", message)
        self.log.see("end")
        self.log.configure(state="disabled")

    def _poll(self):
        try:
            while True:
                kind, pid, value = self.runner.events.get_nowait()
                if self.runner.process is None or self.runner.process.pid != pid:
                    continue
                if kind == "line":
                    self._append(value)
                else:
                    cancelled = self.runner.cancelled
                    self.runner.acknowledge(pid)
                    self.progress.stop()
                    for button in self.task_buttons:
                        button.state(["!disabled"])
                    self.stop_button.state(["disabled"])
                    self.status.set("Stopped." if cancelled else
                                    "Completed — your results are ready." if value == 0 else
                                    "Task could not finish. See the details below.")
                    if value == 0 and self.pending_output:
                        if self.pending_kind == "recording":
                            self.last_recording = self.pending_output
                        else:
                            reports = sorted(self.pending_output.rglob(
                                "quality-gate.html" if self.pending_kind == "quality" else "report.html"))
                            if reports:
                                self.last_report = reports[-1]
                    self.refresh()
        except queue.Empty:
            pass
        self.window.after(100, self._poll)

    def open_file(self, path):
        path = Path(path)
        if not path.exists():
            messagebox.showinfo("Not available yet", "Run the relevant test or recording first.", parent=self.window)
            return
        os.startfile(str(path))

    def open_latest_report(self):
        if self.last_report:
            self.open_file(self.last_report)
            return
        candidates = list((self.root / "artifacts/control-panel").rglob("report.html"))
        candidates.extend((self.root / "artifacts/robustness").rglob("report.html"))
        candidates.extend((self.root / "artifacts").rglob("quality-gate.html"))
        if candidates:
            self.open_file(max(candidates, key=lambda path: path.stat().st_mtime))
        else:
            messagebox.showinfo("No new test yet", "Evaluate a model or run a wind test first.", parent=self.window)

    def open_recordings(self):
        self.open_file(self.last_recording or self.root / "runs")

    def close(self):
        self.runner.close()
        self.window.destroy()


def main():
    window = tk.Tk()
    ControlPanel(window)
    window.mainloop()


if __name__ == "__main__":
    main()

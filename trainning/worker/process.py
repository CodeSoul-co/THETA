from __future__ import annotations

import os
import queue
import re
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence, TextIO

from .errors import JobCancelled, RetryableJobError, ProcessExecutionError


@dataclass(frozen=True)
class ProcessResult:
    return_code: int
    elapsed_seconds: float


CancelCheck = Callable[[], bool]
Heartbeat = Callable[[], None]


class ProcessRunner:
    def __init__(self, poll_seconds: float = 0.5, on_output: Callable[[str], None] | None = None):
        self.poll_seconds = poll_seconds
        self.on_output = on_output

    def run(
        self,
        command: Sequence[str],
        cwd: Path,
        env: Mapping[str, str],
        log_path: Path,
        timeout_seconds: int,
        is_cancelled: CancelCheck,
        heartbeat: Heartbeat,
        shutdown_grace_seconds: int,
    ) -> ProcessResult:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        start = time.monotonic()
        creation_flags = 0
        popen_kwargs: dict[str, object] = {}
        if os.name == "nt":
            creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_kwargs["start_new_session"] = True

        with log_path.open("a", encoding="utf-8", errors="replace") as log:
            self._write_command(log, command)
            process = subprocess.Popen(
                list(command),
                cwd=str(cwd),
                env=dict(env),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creation_flags,
                **popen_kwargs,
            )
            lines: queue.Queue[str | None] = queue.Queue()
            reader = threading.Thread(
                target=self._read_output, args=(process, lines), daemon=True
            )
            reader.start()

            last_heartbeat = 0.0
            try:
                while process.poll() is None:
                    self._drain(lines, log)
                    now = time.monotonic()
                    if is_cancelled():
                        self._terminate(process, shutdown_grace_seconds)
                        raise JobCancelled("用户已请求取消训练任务")
                    if now - start > timeout_seconds:
                        self._terminate(process, shutdown_grace_seconds)
                        raise RetryableJobError(
                            f"训练超过设定的 {timeout_seconds} 秒时限，已停止；请检查计算资源或调整配置"
                        )
                    if now - last_heartbeat >= max(self.poll_seconds, 1.0):
                        heartbeat()
                        last_heartbeat = now
                    time.sleep(self.poll_seconds)
            except BaseException:
                if process.poll() is None:
                    self._terminate(process, shutdown_grace_seconds)
                raise
            finally:
                reader.join(timeout=2)
                self._drain(lines, log)
                if process.stdout is not None:
                    process.stdout.close()

            elapsed = time.monotonic() - start
            if process.returncode != 0:
                with log_path.open('rb') as output:
                    output.seek(max(0, log_path.stat().st_size - 16384))
                    tail = output.read().decode('utf-8', errors='replace')
                causes = re.findall(r'^(?:[\w.]*Error|Exception):[^\r\n]+', tail, re.MULTILINE)
                reason = causes[-1][:1000] if causes else '未捕获具体异常，请查看任务目录中的 worker.log'
                if 'BOW vocabulary is empty' in reason:
                    reason = '正文分词后没有可用词语。请检查正文列、停用词设置，并增加有效文本记录。'
                elif 'Expected more than 1 value per channel' in reason:
                    reason = '有效训练样本不足，当前模型每批至少需要 2 条文本。请增加独立文本记录后重试。'
                for key, value in env.items():
                    if len(value) >= 8 and any(word in key.upper() for word in ('KEY', 'TOKEN', 'SECRET', 'PASSWORD')):
                        reason = reason.replace(value, '[已隐藏]')
                reason = re.sub(r'(?i)(api[_-]?key|authorization|token|secret)([\s=:]+)\S+', r'\1\2[已隐藏]', reason)
                reason = re.sub(r'\bsk-[A-Za-z0-9_-]+', '[已隐藏]', reason)
                raise ProcessExecutionError(f"训练进程异常退出（代码 {process.returncode}）：{reason}")
            return ProcessResult(process.returncode, elapsed)

    @staticmethod
    def _read_output(process: subprocess.Popen[str], lines: queue.Queue[str | None]) -> None:
        assert process.stdout is not None
        try:
            for line in process.stdout:
                lines.put(line)
        finally:
            lines.put(None)

    def _drain(self, lines: queue.Queue[str | None], log: TextIO) -> None:
        wrote = False
        while True:
            try:
                line = lines.get_nowait()
            except queue.Empty:
                break
            if line is None:
                continue
            log.write(line)
            if self.on_output:
                self.on_output(line)
            wrote = True
        if wrote:
            log.flush()

    def _write_command(self, log: TextIO, command: Sequence[str]) -> None:
        # The command contains only validated task values, but use repr to keep logs unambiguous.
        log.write("COMMAND: " + " ".join(repr(part) for part in command) + "\n")
        log.flush()
        if self.on_output:
            self.on_output("COMMAND:")

    @staticmethod
    def _terminate(process: subprocess.Popen[str], grace_seconds: int) -> None:
        try:
            if os.name == "nt":
                process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=grace_seconds)
            return
        except (ProcessLookupError, subprocess.TimeoutExpired, OSError):
            pass
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return

"""PTY regression: real ./theta UI and local status transport, synthetic job, no training/inference."""
import codecs
import fcntl
import json
import os
from pathlib import Path
import pty
import select
import sqlite3
import struct
import sys
import subprocess
import tempfile
import termios
import time

# Test-only dependency: python -m pip install pyte (not part of the Agent runtime).
import pyte

REPO = Path(__file__).resolve().parents[2]
COLUMNS = int(sys.argv[1]) if len(sys.argv) > 1 else 80
with tempfile.TemporaryDirectory(prefix='theta-monitor-') as folder:
    home = Path(folder)
    job_id = 'job-' + 'c' * 64
    session_id = 'chat-monitor-fixture'
    session = dict(id=session_id, title='状态监控回归测试', runId='run-monitor', monitorTraining=True,
                   datasetRefs=[], messages=[{'role': 'assistant', 'content': '历史记录保留标记\n用户研究目标已经保存，训练需要用户确认。'}], updatedAt='2026-09-05T00:00:00Z')
    with sqlite3.connect(home / 'sessions.sqlite') as db:
        db.execute('CREATE TABLE sessions (id TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL, lease TEXT, expires INTEGER)')
        db.execute('INSERT INTO sessions VALUES (?,?,?,NULL,NULL)', (session_id, json.dumps(session), session['updatedAt']))
    with sqlite3.connect(home / 'research.sqlite') as db:
        db.execute('CREATE TABLE records (kind TEXT, id TEXT, value TEXT, PRIMARY KEY(kind,id))')
        run = dict(id='run-monitor', goal='测试状态展示', jobs=[job_id], activeJob=job_id, notes=[], computeBackend='local')
        db.execute('INSERT INTO records VALUES (?,?,?)', ('run', run['id'], json.dumps(run)))
    with sqlite3.connect(home / 'compute.sqlite') as db:
        db.execute('CREATE TABLE jobs (id TEXT PRIMARY KEY, request TEXT NOT NULL, state TEXT NOT NULL, value TEXT NOT NULL, cancel INTEGER NOT NULL DEFAULT 0, updated REAL NOT NULL)')
    job = dict(id=job_id, status='running', phase='training', percent=60, startedAt=time.time()-60, phaseStartedAt=time.time()-30)
    def save_job(heartbeat_age=0):
        with sqlite3.connect(home / 'compute.sqlite') as db:
            db.execute('INSERT OR REPLACE INTO jobs VALUES (?,?,?,?,0,?)', (job_id, '{}', job['status'], json.dumps(job), time.time()-heartbeat_age))
    save_job()
    log = home / 'compute' / job_id / 'jobs/task-1-attempt-1/worker.log'
    log.parent.mkdir(parents=True)
    log.write_text('iteration: 1 of max_iter: 50\n')
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack('HHHH', 40, COLUMNS, 0, 0))
    env = {k: v for k, v in os.environ.items() if not any(x in k for x in ['DEEPSEEK', 'OPENAI', 'MINIMAX', 'THETA_', 'ANTHROPIC'])}
    env.update(TERM='xterm-256color', NO_COLOR='1', THETA_AGENT_HOME=folder, THETA_ENV_FILE=str(home/'no-env'))
    child = subprocess.Popen([str(REPO/'theta'), '--session', session_id], cwd=REPO, env=env, stdin=slave, stdout=slave, stderr=slave)
    os.close(slave)
    output = bytearray()
    screen = pyte.HistoryScreen(COLUMNS, 40, history=2000)
    stream = pyte.Stream(screen)
    decoder = codecs.getincrementaldecoder('utf-8')()
    def visible():
        history = [''.join(line[x].data for x in range(screen.columns)) for line in screen.history.top]
        return '\n'.join([*history, *screen.display])
    def preserved():
        text = visible()
        assert '历史记录保留标记' in text, 'Heartbeat erased transcript: '+text
        assert 'Enter 发送' in text, 'Heartbeat erased composer: '+text

    def until(text, timeout=10):
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            if text in output.decode(errors='replace') and not select.select([master], [], [], .05)[0]:
                return
            if select.select([master], [], [], .1)[0]:
                try:
                    chunk = os.read(master, 65536)
                    if not chunk:
                        if text in output.decode(errors='replace'): return
                        break
                    output.extend(chunk); stream.feed(decoder.decode(chunk))
                except OSError:
                    if text in output.decode(errors='replace'): return
                    break
        raise AssertionError('Missing output: '+text+'\n'+output.decode(errors='replace')[-1800:])
    try:
        until('已报告迭代 1/50')
        os.write(master, b' ' * 100 + b'/hep\x1b[D')  # Wrapped input; cursor before the final p.
        for iteration in range(2, 7):
            log.write_text(f'iteration: {iteration} of max_iter: 50\n'); save_job()
            until(f'已报告迭代 {iteration}/50')
            preserved()
            assert any(line.rstrip().endswith('/hep') for line in screen.display), 'Wrapped input missing from rendered screen'
        save_job(heartbeat_age=60)
        until('超过30秒未收到心跳'); preserved()
        # Force one failed read, then recover. No new job may be submitted.
        with sqlite3.connect(home/'compute.sqlite') as db: db.execute('DELETE FROM jobs')
        until('状态查询失败'); preserved()
        save_job(); log.write_text('iteration: 7 of max_iter: 50\n')
        until('已报告迭代 7/50'); preserved()
        result_dir = home / 'completed result'
        result_dir.mkdir()
        job.update(status='completed', phase='completed', percent=100, resultDir=str(result_dir))
        save_job()
        until('训练结果目录'); until(str(result_dir)); preserved()
        os.write(master, b'l\n')  # Inserts before p: the complete command must still be /help.
        until('/attach "文件路径"')  # /help survived all intermediate redraws.
        os.write(master, b'/exit\n')
        until('对话已保存')
        assert child.wait(timeout=5) == 0
        with sqlite3.connect(home/'compute.sqlite') as db:
            assert db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0] == 1
        print('PASS: real ./theta PTY; iteration refresh, stale heartbeat, failed query recovery, rendered transcript/composer and partial input preserved across repeated redraws; completed result directory printed without inference; no training or inference.')
    finally:
        if child.poll() is None: child.terminate(); child.wait(timeout=5)
        os.close(master)

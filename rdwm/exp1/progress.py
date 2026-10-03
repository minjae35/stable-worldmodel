"""Multi-line progress display that stays on in a tmux pane.

tqdm turns itself off when stdout is not a terminal. This writer always
draws, and it rewrites the same four lines with ANSI cursor moves when the
output is a TTY (a tmux pane is). A plain one-line copy is appended to
``log_path`` so the log stays readable.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path


def _clock(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f'{h}h{m:02d}m{s:02d}s'
    return f'{m}m{s:02d}s'


class Progress:
    def __init__(self, total: int, initial: int, desc: str, log_path: Path | None, unit: str = 'it/s'):
        self.total = int(total)
        self.n = int(initial)
        self.initial = int(initial)
        self.desc = desc
        self.unit = unit
        self.postfix = ''
        self.t0 = time.time()
        self._lines = 0
        self._out = sys.__stdout__
        self._tty = bool(self._out.isatty())
        self._log = open(log_path, 'a') if log_path else None

    def set_postfix(self, text: str) -> None:
        self.postfix = text

    def update(self, step: int | None = None) -> None:
        self.n = self.n + 1 if step is None else int(step)
        self._draw()

    def close(self) -> None:
        if self._lines:
            self._out.write('\n')
            self._out.flush()
        if self._log:
            self._log.close()
            self._log = None

    def _draw(self) -> None:
        done = self.n - self.initial
        elapsed = time.time() - self.t0
        rate = done / elapsed if elapsed > 0 and done > 0 else 0.0
        eta = (self.total - self.n) / rate if rate > 0 else 0.0
        frac = self.n / self.total if self.total else 1.0
        width = 20
        filled = min(width, int(round(width * frac)))
        bar = '█' * filled + '-' * (width - filled)
        lines = [
            self.desc,
            f'{self.n}/{self.total} [{bar}] {100 * frac:.0f}%',
            f'elapsed {_clock(elapsed)} | ETA {_clock(eta)} | {rate:.2f} {self.unit}',
            self.postfix,
        ]
        if self._tty and self._lines:
            self._out.write(f'\x1b[{self._lines}A')
        for line in lines:
            if self._tty:
                self._out.write('\x1b[2K' + line + '\n')
            else:
                self._out.write(line + '\n')
        self._out.flush()
        self._lines = len(lines) if self._tty else 0
        if self._log:
            self._log.write(' | '.join(lines) + '\n')
            self._log.flush()

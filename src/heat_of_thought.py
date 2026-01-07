"""
heat_of_thought.py - Real-time State Heatmap Visualizer

Shows the "Heat of Thought" - a live visualization of the 2KB persistent state
as data flows through the Liquid Mamba model.

Data sources: sine wave, random noise, text file, or keyboard input.
"""

import curses
import time
import numpy as np
import os
import sys
from collections import deque

D_INNER = 128
D_STATE = 16

lfsr_state = 0xACE1

def lfsr_next():
    global lfsr_state
    bit = ((lfsr_state >> 0) ^ (lfsr_state >> 2) ^
           (lfsr_state >> 3) ^ (lfsr_state >> 5)) & 1
    lfsr_state = ((lfsr_state >> 1) | (bit << 15)) & 0xFFFF
    return lfsr_state

def init_dt_lut():
    lut = np.zeros(256, dtype=np.int8)
    for i in range(256):
        x = (i - 128) / 128.0
        dt = 0.001 + 0.099 * (x + 1.0) / 2.0
        lut[i] = int(np.exp(-dt) * 127)
    return lut

class DataSource:
    """Generates input bytes from various sources."""
    def __init__(self, mode='sine', file_path=None):
        self.mode = mode
        self.t = 0
        self.file_data = None
        self.file_pos = 0
        self.last_key = 0  # For keyboard mode

        if mode == 'file' and file_path:
            try:
                with open(file_path, 'rb') as f:
                    self.file_data = f.read()
            except:
                self.mode = 'sine'

    def set_key(self, key):
        """Set the last pressed key for keyboard mode."""
        self.last_key = key - 128 if key >= 0 else 0

    def next_byte(self):
        self.t += 1

        if self.mode == 'sine':
            return int(80 * np.sin(self.t * 0.05))

        elif self.mode == 'random':
            return int(np.random.randint(-80, 80))

        elif self.mode == 'mixed':
            phase = (self.t // 500) % 4
            if phase == 0:
                return int(80 * np.sin(self.t * 0.05))
            elif phase == 1:
                return int(np.random.randint(-80, 80))
            elif phase == 2:
                return int(40 * np.sin(self.t * 0.2) + 40 * np.sin(self.t * 0.01))
            else:
                return int(80 * np.sign(np.sin(self.t * 0.1)))

        elif self.mode == 'keyboard':
            # Decay the key value over time for visual effect
            decay = max(0, abs(self.last_key) - self.t % 10)
            return int(np.sign(self.last_key) * decay) if self.last_key != 0 else 0

        elif self.mode == 'file' and self.file_data:
            byte = self.file_data[self.file_pos % len(self.file_data)]
            self.file_pos += 1
            return int(byte) - 128

        else:
            return int(80 * np.sin(self.t * 0.05))

class LiquidSSM:
    def __init__(self):
        self.A = np.full((D_INNER, D_STATE), -1, dtype=np.int8)
        self.B = np.full(D_STATE, 80, dtype=np.int8)
        self.h = np.zeros((D_INNER, D_STATE), dtype=np.float32)
        self.dt_lut = init_dt_lut()
        self.last_dt = 0.5

    def step(self, x_byte):
        dt_idx = int(128 + np.sin(x_byte * 0.1) * 50) & 0xFF
        dt_scale = self.dt_lut[dt_idx]
        self.last_dt = (dt_idx - 128) / 128.0 * 0.5 + 0.5

        for d in range(D_INNER):
            for n in range(D_STATE):
                decay = 0.75 + 0.1 * ((d + n) % 5) / 5.0
                h_input = 0.3 * (20 + (n % 10)) * x_byte / 127.0
                self.h[d, n] = decay * self.h[d, n] + h_input
                self.h[d, n] = max(-127, min(127, self.h[d, n]))

        return self.h.copy(), self.last_dt

def main(stdscr):
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_WHITE, -1)
    curses.init_pair(2, curses.COLOR_BLUE, -1)
    curses.init_pair(3, curses.COLOR_GREEN, -1)
    curses.init_pair(4, curses.COLOR_YELLOW, -1)
    curses.init_pair(5, curses.COLOR_RED, -1)
    curses.init_pair(6, curses.COLOR_CYAN, -1)
    curses.init_pair(7, curses.COLOR_MAGENTA, -1)

    curses.curs_set(0)
    stdscr.nodelay(True)

    max_y, max_x = stdscr.getmaxyx()
    ssm = LiquidSSM()

    # Data source modes
    modes = ['sine', 'random', 'mixed', 'keyboard']
    mode_names = {
        'sine': 'Sine Wave',
        'random': 'Random Noise',
        'mixed': 'Mixed Patterns',
        'keyboard': 'KEYBOARD (type!)'
    }
    mode_idx = 0

    # Check for file argument
    file_path = None
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        file_path = sys.argv[1]
        modes.insert(0, 'file')
        mode_names['file'] = f'File: {os.path.basename(file_path)}'

    source = DataSource(modes[mode_idx], file_path)

    bytes_processed = 0
    start_time = time.time()

    energy_history = deque(maxlen=30)
    dt_history = deque(maxlen=30)

    EXPLAIN = [
        ("CONTROLS:", 6, True),
        ("  [1-4] Switch data source", 1, False),
        ("  [q] Quit", 1, False),
        ("", 1, False),
        ("HEATMAP: Brain state (2KB)", 3, False),
        ("  · dim  █ active", 1, False),
        ("", 1, False),
        ("Δ: Thinking speed", 3, False),
        ("  Blue=fast  Red=complex", 1, False),
        ("", 1, False),
        ("O(1) constant time!", 6, False),
    ]

    while True:
        try:
            key = stdscr.getch()
            if key == ord('q'):
                break
            elif key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5')]:
                mode_idx = min(key - ord('1'), len(modes) - 1)
                source = DataSource(modes[mode_idx], file_path)
            elif key > 0 and source.mode == 'keyboard':
                # Feed keystroke to the model
                source.set_key(key)
        except:
            pass

        for _ in range(30):
            x = source.next_byte()
            state, dt = ssm.step(x)
            bytes_processed += 1

        elapsed = time.time() - start_time
        rate = bytes_processed / elapsed if elapsed > 0 else 0
        energy = np.sum(np.abs(ssm.h))

        energy_history.append(energy)
        dt_history.append(dt)

        stdscr.clear()

        hm_cols = min(32, max_x // 2 - 4)
        hm_rows = min(12, max_y - 12)
        explain_x = hm_cols + 6

        try:
            # Title with mode
            stdscr.addstr(0, 0, "LIQUID ANALOG STREAM", curses.color_pair(6) | curses.A_BOLD)
            stdscr.addstr(1, 0, f"Mode: {mode_names[modes[mode_idx]]}", curses.color_pair(7))

            # Stats
            dt_color = 2 if dt < 0.3 else (3 if dt < 0.6 else 5)
            stdscr.addstr(2, 0, f"Bytes: {bytes_processed:,}  Rate: {rate:,.0f}/s  Energy: {energy:,.0f}", curses.color_pair(dt_color))

            # Heatmap
            stdscr.addstr(4, 0, "BRAIN STATE:", curses.color_pair(6) | curses.A_BOLD)

            flat = ssm.h.flatten()[:hm_rows * hm_cols]
            for i, val in enumerate(flat):
                y = 5 + (i // hm_cols)
                x_pos = 1 + (i % hm_cols)
                if y < max_y - 6 and x_pos < max_x - 1:
                    intensity = abs(val) / 127.0
                    if intensity < 0.1:
                        char, pair = ' ', 1
                    elif intensity < 0.2:
                        char, pair = '·', 1
                    elif intensity < 0.4:
                        char, pair = '░', 2
                    elif intensity < 0.6:
                        char, pair = '▒', 3
                    elif intensity < 0.8:
                        char, pair = '▓', 4
                    else:
                        char, pair = '█', 5
                    stdscr.addch(y, x_pos, char, curses.color_pair(pair))

            # Delta bar
            bar_y = 5 + hm_rows + 1
            dt_label = "Fast" if dt < 0.3 else ("Med" if dt < 0.6 else "Slow")
            stdscr.addstr(bar_y, 0, f"Δ [{dt_label:4}]:", curses.color_pair(dt_color) | curses.A_BOLD)
            bar_len = min(20, hm_cols - 1)
            bar = '█' * int(dt * bar_len) + '░' * (bar_len - int(dt * bar_len))
            stdscr.addstr(bar_y, 12, bar, curses.color_pair(dt_color))

            # Energy waveform
            wave_y = bar_y + 2
            if wave_y < max_y - 2 and len(energy_history) > 1:
                stdscr.addstr(wave_y, 0, "ENERGY:", curses.color_pair(6) | curses.A_BOLD)
                e_min = min(energy_history)
                e_max = max(energy_history) + 1
                wave_chars = " ▁▂▃▄▅▆▇█"
                for i, (e, d) in enumerate(zip(energy_history, dt_history)):
                    if i < hm_cols:
                        normalized = (e - e_min) / (e_max - e_min)
                        char_idx = int(normalized * (len(wave_chars) - 1))
                        d_color = 2 if d < 0.3 else (3 if d < 0.6 else 5)
                        stdscr.addch(wave_y + 1, 1 + i, wave_chars[char_idx], curses.color_pair(d_color))

            # Explanation panel
            if max_x > 55:
                exp_y = 4
                for text, color, bold in EXPLAIN:
                    if exp_y < max_y - 2:
                        attr = curses.color_pair(color)
                        if bold:
                            attr |= curses.A_BOLD
                        stdscr.addstr(exp_y, explain_x, text[:max_x - explain_x - 1], attr)
                        exp_y += 1

            stdscr.addstr(max_y - 1, 0, "[1-4] modes  [q] quit", curses.color_pair(1))

        except curses.error:
            pass

        stdscr.refresh()
        time.sleep(0.05)

if __name__ == "__main__":
    print("LIQUID ANALOG STREAM - Heat of Thought")
    print("Usage: python3 heat_of_thought.py [file_path]")
    print("Controls: [1-4] switch modes, [q] quit")
    time.sleep(0.5)
    curses.wrapper(main)

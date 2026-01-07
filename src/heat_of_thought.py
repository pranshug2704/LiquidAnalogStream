"""
heat_of_thought.py - Real-time State Heatmap Visualizer

Shows the "Heat of Thought" - a live visualization of the 2KB persistent state
as data flows through the Liquid Mamba model.

Features:
- 32×64 heatmap of state values
- Live Δ (viscosity) bar with color coding
- Throughput stats
- Input stream display
"""

import curses
import time
import numpy as np
import sys
import threading
from queue import Queue

# SSM Configuration
D_INNER = 128
D_STATE = 16
HEATMAP_ROWS = 32
HEATMAP_COLS = 64

# LFSR for stochastic rounding
lfsr_state = 0xACE1

def lfsr_next():
    global lfsr_state
    bit = ((lfsr_state >> 0) ^ (lfsr_state >> 2) ^
           (lfsr_state >> 3) ^ (lfsr_state >> 5)) & 1
    lfsr_state = ((lfsr_state >> 1) | (bit << 15)) & 0xFFFF
    return lfsr_state

def stochastic_round(val, shift=7):
    mask = (1 << shift) - 1
    fraction = int(val) & mask
    threshold = lfsr_next() & mask
    rounded = int(val) + ((1 << shift) if fraction > threshold else 0)
    rounded >>= shift
    return max(-127, min(127, rounded))

def init_dt_lut():
    lut = np.zeros(256, dtype=np.int8)
    for i in range(256):
        x = (i - 128) / 128.0
        dt = 0.001 + 0.099 * (x + 1.0) / 2.0
        lut[i] = int(np.exp(-dt) * 127)
    return lut

class LiquidSSM:
    def __init__(self):
        self.A = np.full((D_INNER, D_STATE), -3, dtype=np.int8)
        self.B = np.full(D_STATE, 25, dtype=np.int8)
        self.C = np.full(D_STATE, 25, dtype=np.int8)
        self.D = np.full(D_INNER, 64, dtype=np.int8)
        self.h = np.zeros((D_INNER, D_STATE), dtype=np.int8)
        self.dt_lut = init_dt_lut()
        self.last_dt = 0.05

    def step(self, x_byte):
        # Compute input-dependent dt (the "Liquid" part)
        dt_idx = int(128 + np.sin(x_byte * 0.1) * 50) & 0xFF
        dt_scale = self.dt_lut[dt_idx]
        self.last_dt = (dt_idx - 128) / 128.0 * 0.5 + 0.5  # Normalize to [0, 1]

        for d in range(D_INNER):
            for n in range(D_STATE):
                A_abs = abs(self.A[d, n])
                dA = (int(dt_scale) * A_abs) >> 7
                dA = max(1, min(127, dA))
                dB = (int(dt_scale) * int(self.B[n])) >> 7

                h_new = int(dA) * int(self.h[d, n]) + int(dB) * int(x_byte)
                self.h[d, n] = stochastic_round(h_new)

        return self.h.copy(), self.last_dt

def draw_heatmap(win, state, start_y, start_x):
    """Draw the 2D heatmap of state values."""
    # Reshape 128×16 to 32×64 for display
    flat = state.flatten()
    reshaped = flat.reshape(HEATMAP_ROWS, HEATMAP_COLS)

    # Color pairs: 1=dim, 2=blue, 3=green, 4=yellow, 5=red
    for y in range(HEATMAP_ROWS):
        for x in range(HEATMAP_COLS):
            val = reshaped[y, x]
            intensity = abs(val) / 127.0

            if intensity < 0.2:
                char, pair = '·', 1
            elif intensity < 0.4:
                char, pair = '░', 2
            elif intensity < 0.6:
                char, pair = '▒', 3
            elif intensity < 0.8:
                char, pair = '▓', 4
            else:
                char, pair = '█', 5

            try:
                win.addch(start_y + y, start_x + x, char, curses.color_pair(pair))
            except:
                pass

def draw_dt_bar(win, dt, y, x, width=40):
    """Draw the delta (viscosity) bar."""
    # dt is 0-1, where 0=fast(blue) and 1=slow(red)
    filled = int(dt * width)

    # Determine color based on dt
    if dt < 0.3:
        pair = 2  # Blue - fast/easy
    elif dt < 0.6:
        pair = 3  # Green - medium
    else:
        pair = 5  # Red - slow/complex

    try:
        bar = '█' * filled + '░' * (width - filled)
        win.addstr(y, x, bar, curses.color_pair(pair))
    except:
        pass

def main(stdscr):
    # Setup colors
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_WHITE, -1)     # Dim
    curses.init_pair(2, curses.COLOR_BLUE, -1)      # Cool/fast
    curses.init_pair(3, curses.COLOR_GREEN, -1)     # Medium
    curses.init_pair(4, curses.COLOR_YELLOW, -1)    # Warm
    curses.init_pair(5, curses.COLOR_RED, -1)       # Hot/slow
    curses.init_pair(6, curses.COLOR_CYAN, -1)      # Title

    curses.curs_set(0)
    stdscr.nodelay(True)

    # Initialize
    ssm = LiquidSSM()

    # Stats
    bytes_processed = 0
    start_time = time.time()
    recent_bytes = []

    # Generate input data (sine wave pattern)
    t = 0

    while True:
        # Check for quit
        try:
            key = stdscr.getch()
            if key == ord('q'):
                break
        except:
            pass

        # Process a batch of bytes
        batch_size = 100
        for _ in range(batch_size):
            t += 1
            x = int(80 * np.sin(t * 0.05))
            state, dt = ssm.step(x)
            bytes_processed += 1
            recent_bytes.append(x)
            if len(recent_bytes) > 20:
                recent_bytes.pop(0)

        # Calculate stats
        elapsed = time.time() - start_time
        rate = bytes_processed / elapsed if elapsed > 0 else 0
        energy = np.sum(np.abs(ssm.h))

        # Clear and redraw
        stdscr.clear()

        # Header
        title = "╔═══════════════════════════════════════════════════════════════════════╗"
        stdscr.addstr(0, 0, title, curses.color_pair(6))
        stdscr.addstr(1, 0, "║", curses.color_pair(6))
        stdscr.addstr(1, 2, "LIQUID ANALOG STREAM - Heat of Thought", curses.color_pair(6) | curses.A_BOLD)
        stdscr.addstr(1, 73, "║", curses.color_pair(6))
        stdscr.addstr(2, 0, "╠═══════════════════════════════════════════════════════════════════════╣", curses.color_pair(6))

        # Stats line
        stats = f"│ Bytes: {bytes_processed:>10,} │ Rate: {rate:>10,.0f}/s │ Energy: {energy:>8,.0f} │ Press 'q' to quit │"
        stdscr.addstr(3, 0, stats, curses.color_pair(1))
        stdscr.addstr(4, 0, "╠═══════════════════════════════════════════════════════════════════════╣", curses.color_pair(6))

        # Heatmap label
        stdscr.addstr(5, 2, "State Heatmap (128×16 = 2KB):", curses.color_pair(1) | curses.A_BOLD)

        # Draw heatmap (starting at row 6)
        draw_heatmap(stdscr, ssm.h, 6, 4)

        # Delta bar
        stdscr.addstr(39, 2, "Δ (Viscosity):", curses.color_pair(1) | curses.A_BOLD)
        dt_label = "Fast" if dt < 0.3 else ("Med" if dt < 0.6 else "Slow")
        stdscr.addstr(39, 17, f"[{dt_label}]", curses.color_pair(2 if dt < 0.3 else (3 if dt < 0.6 else 5)))
        draw_dt_bar(stdscr, dt, 40, 2, 50)

        # Recent input
        stdscr.addstr(42, 2, "Input Stream:", curses.color_pair(1) | curses.A_BOLD)
        stream_str = ' '.join([f"{b:+4d}" for b in recent_bytes[-10:]])
        stdscr.addstr(43, 2, stream_str, curses.color_pair(3))

        # Footer
        stdscr.addstr(45, 0, "╚═══════════════════════════════════════════════════════════════════════╝", curses.color_pair(6))

        stdscr.refresh()
        time.sleep(0.016)  # ~60fps

if __name__ == "__main__":
    print("Starting Heat of Thought visualizer...")
    print("Press 'q' to quit")
    time.sleep(1)
    curses.wrapper(main)

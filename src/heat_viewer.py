"""
heat_viewer.py - Async State Viewer (Consumer)

Reads state from shared memory and renders at 60fps.
Run liquid_producer in parallel for full-speed inference.

Usage:
  Terminal 1: ./hardware/liquid_producer [file]
  Terminal 2: python3 src/heat_viewer.py
"""

import curses
import time
import struct
import mmap
import os
from collections import deque

D_INNER = 128
D_STATE = 16
STATE_SIZE = D_INNER * D_STATE

# SharedState struct layout (must match C++)
# uint64_t bytes_processed (8)
# uint64_t timestamp_us (8)
# float dt (4)
# float energy (4)
# int8_t state[128][16] (2048)
HEADER_SIZE = 8 + 8 + 4 + 4  # 24 bytes
TOTAL_SIZE = HEADER_SIZE + STATE_SIZE  # 2072 bytes

def read_shared_state(mm):
    """Read state from mmap."""
    try:
        mm.seek(0)
        data = mm.read(TOTAL_SIZE)
        if len(data) < TOTAL_SIZE:
            return None

        bytes_processed, timestamp_us, dt, energy = struct.unpack('<QQff', data[:HEADER_SIZE])
        state = list(struct.unpack(f'<{STATE_SIZE}b', data[HEADER_SIZE:]))

        return {
            'bytes': bytes_processed,
            'timestamp': timestamp_us,
            'dt': dt,
            'energy': energy,
            'state': state
        }
    except:
        return None

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

    # Open shared memory
    shm_path = "/tmp/liquid_state.bin"
    if not os.path.exists(shm_path):
        stdscr.addstr(0, 0, f"Waiting for producer... ({shm_path})", curses.color_pair(5))
        stdscr.refresh()
        while not os.path.exists(shm_path):
            time.sleep(0.1)
            try:
                if stdscr.getch() == ord('q'):
                    return
            except:
                pass

    fd = os.open(shm_path, os.O_RDONLY)
    mm = mmap.mmap(fd, TOTAL_SIZE, access=mmap.ACCESS_READ)

    energy_history = deque(maxlen=40)
    dt_history = deque(maxlen=40)
    last_bytes = 0
    last_time = time.time()
    rate = 0

    while True:
        try:
            key = stdscr.getch()
            if key == ord('q'):
                break
        except:
            pass

        # Read state from shared memory
        state = read_shared_state(mm)
        if not state:
            time.sleep(0.01)
            continue

        # Calculate rate
        now = time.time()
        elapsed = now - last_time
        if elapsed > 0.1:
            rate = (state['bytes'] - last_bytes) / elapsed
            last_bytes = state['bytes']
            last_time = now

        dt = state['dt']
        energy = state['energy']

        energy_history.append(energy)
        dt_history.append(dt)

        stdscr.clear()

        hm_cols = min(32, max_x // 2 - 4)
        hm_rows = min(12, max_y - 12)
        explain_x = hm_cols + 8

        try:
            # Title
            stdscr.addstr(0, 0, "LIQUID ANALOG STREAM", curses.color_pair(6) | curses.A_BOLD)
            stdscr.addstr(0, 22, " - Async Viewer", curses.color_pair(7))

            # Stats
            dt_color = 2 if dt < 0.3 else (3 if dt < 0.6 else 5)
            rate_str = f"{rate/1000000:.1f}M/s" if rate > 1000000 else f"{rate/1000:.0f}K/s"
            stdscr.addstr(1, 0, f"Bytes: {state['bytes']:,}", curses.color_pair(1))
            stdscr.addstr(1, 25, f"Rate: {rate_str}", curses.color_pair(3) | curses.A_BOLD)
            stdscr.addstr(1, 45, f"Energy: {energy:,.0f}", curses.color_pair(dt_color))

            # Heatmap
            stdscr.addstr(3, 0, "BRAIN STATE:", curses.color_pair(6) | curses.A_BOLD)

            for i in range(min(len(state['state']), hm_rows * hm_cols)):
                y = 4 + (i // hm_cols)
                x_pos = 1 + (i % hm_cols)
                if y < max_y - 6 and x_pos < max_x - 1:
                    val = state['state'][i]
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
            bar_y = 4 + hm_rows + 1
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

            # Info panel
            if max_x > 55:
                info = [
                    ("ASYNC MODE", 7, True),
                    ("", 1, False),
                    ("Producer: C++ NEON", 3, False),
                    ("Consumer: Python UI", 2, False),
                    ("", 1, False),
                    ("Shared via mmap", 1, False),
                    ("No blocking!", 6, False),
                ]
                exp_y = 4
                for text, color, bold in info:
                    if exp_y < max_y - 2:
                        attr = curses.color_pair(color)
                        if bold:
                            attr |= curses.A_BOLD
                        stdscr.addstr(exp_y, explain_x, text, attr)
                        exp_y += 1

            stdscr.addstr(max_y - 1, 0, "[q] quit | Run liquid_producer in another terminal", curses.color_pair(1))

        except curses.error:
            pass

        stdscr.refresh()
        time.sleep(0.016)  # 60fps

    mm.close()
    os.close(fd)

if __name__ == "__main__":
    print("Liquid Viewer - Async State Visualizer")
    print("Run ./hardware/liquid_producer in another terminal first!")
    time.sleep(1)
    curses.wrapper(main)

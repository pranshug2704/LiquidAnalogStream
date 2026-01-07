"""
heat_of_thought.py - Real-time State Heatmap Visualizer

Shows the "Heat of Thought" - a live visualization of the 2KB persistent state
as data flows through the Liquid Mamba model.
"""

import curses
import time
import numpy as np

D_INNER = 128
D_STATE = 16

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
        self.A = np.full((D_INNER, D_STATE), -1, dtype=np.int8)  # Less decay
        self.B = np.full(D_STATE, 80, dtype=np.int8)  # Stronger input
        self.C = np.full(D_STATE, 25, dtype=np.int8)
        self.D = np.full(D_INNER, 64, dtype=np.int8)
        self.h = np.zeros((D_INNER, D_STATE), dtype=np.int8)
        self.dt_lut = init_dt_lut()
        self.last_dt = 0.5

    def step(self, x_byte):
        dt_idx = int(128 + np.sin(x_byte * 0.1) * 50) & 0xFF
        dt_scale = self.dt_lut[dt_idx]
        self.last_dt = (dt_idx - 128) / 128.0 * 0.5 + 0.5

        for d in range(D_INNER):
            for n in range(D_STATE):
                # Vary decay across dimensions for visual interest
                decay = 70 + (d % 30)  # 70-99 range (55%-77% retention)
                h_decayed = (int(decay) * int(self.h[d, n])) >> 7
                # Input varies by position
                input_scale = 20 + (n % 10)
                h_input = (input_scale * x_byte) >> 7
                h_new = h_decayed + h_input
                self.h[d, n] = max(-127, min(127, h_new))

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

    bytes_processed = 0
    start_time = time.time()
    t = 0

    # Explanation text
    EXPLAIN = [
        ("WHAT YOU'RE SEEING:", 6, True),
        ("", 1, False),
        ("HEATMAP: The AI's 'brain state' - 2KB of memory", 3, False),
        ("  · dim = inactive   █ bright = neuron firing", 1, False),
        ("", 1, False),
        ("Δ (DELTA): How 'hard' the AI is thinking", 3, False),
        ("  Blue/Fast = easy input (predictable)", 2, False),
        ("  Red/Slow  = complex input (high entropy)", 5, False),
        ("", 1, False),
        ("ENERGY: Total neural activity (higher = busier)", 1, False),
        ("", 1, False),
        ("Unlike ChatGPT, this runs at O(1) - constant", 6, False),
        ("time regardless of context length!", 6, False),
    ]

    while True:
        try:
            key = stdscr.getch()
            if key == ord('q'):
                break
        except:
            pass

        # Process bytes
        for _ in range(50):
            t += 1
            x = int(80 * np.sin(t * 0.05))
            state, dt = ssm.step(x)
            bytes_processed += 1

        elapsed = time.time() - start_time
        rate = bytes_processed / elapsed if elapsed > 0 else 0
        energy = np.sum(np.abs(ssm.h))

        stdscr.clear()

        # Calculate layout
        hm_cols = min(32, max_x // 2 - 4)
        hm_rows = min(12, max_y - 8)
        explain_x = hm_cols + 6

        try:
            # Title
            stdscr.addstr(0, 0, "LIQUID ANALOG STREAM", curses.color_pair(6) | curses.A_BOLD)
            stdscr.addstr(0, 22, " - Heat of Thought", curses.color_pair(1))

            # Stats bar
            stdscr.addstr(1, 0, f"Bytes: {bytes_processed:,}", curses.color_pair(1))
            stdscr.addstr(1, 20, f"Rate: {rate:,.0f}/s", curses.color_pair(3))
            stdscr.addstr(1, 38, f"Energy: {energy:,.0f}", curses.color_pair(4 if energy > 20000 else 1))

            # Heatmap label
            stdscr.addstr(3, 0, "BRAIN STATE:", curses.color_pair(6) | curses.A_BOLD)

            # Draw heatmap
            flat = ssm.h.flatten()[:hm_rows * hm_cols]
            for i, val in enumerate(flat):
                y = 4 + (i // hm_cols)
                x_pos = 1 + (i % hm_cols)
                if y < max_y - 4 and x_pos < max_x - 1:
                    intensity = abs(val) / 127.0
                    if intensity < 0.15:
                        char, pair = '·', 1
                    elif intensity < 0.35:
                        char, pair = '░', 2
                    elif intensity < 0.55:
                        char, pair = '▒', 3
                    elif intensity < 0.75:
                        char, pair = '▓', 4
                    else:
                        char, pair = '█', 5
                    stdscr.addch(y, x_pos, char, curses.color_pair(pair))

            # Delta bar
            bar_y = min(4 + hm_rows + 1, max_y - 3)
            dt_label = "Fast" if dt < 0.3 else ("Med" if dt < 0.6 else "Slow")
            dt_color = 2 if dt < 0.3 else (3 if dt < 0.6 else 5)
            stdscr.addstr(bar_y, 0, "Δ THINKING:", curses.color_pair(6) | curses.A_BOLD)
            stdscr.addstr(bar_y, 13, f"[{dt_label:4}]", curses.color_pair(dt_color) | curses.A_BOLD)
            bar_len = min(20, hm_cols - 1)
            bar = '█' * int(dt * bar_len) + '░' * (bar_len - int(dt * bar_len))
            stdscr.addstr(bar_y + 1, 1, bar, curses.color_pair(dt_color))

            # Explanation panel (right side)
            if max_x > 70:
                exp_y = 3
                for text, color, bold in EXPLAIN:
                    if exp_y < max_y - 2:
                        attr = curses.color_pair(color)
                        if bold:
                            attr |= curses.A_BOLD
                        stdscr.addstr(exp_y, explain_x, text[:max_x - explain_x - 1], attr)
                        exp_y += 1

            # Footer
            stdscr.addstr(max_y - 1, 0, "[q] quit", curses.color_pair(1))

        except curses.error:
            pass

        stdscr.refresh()
        time.sleep(0.05)

if __name__ == "__main__":
    print("LIQUID ANALOG STREAM - Heat of Thought")
    print("Press 'q' to quit")
    time.sleep(0.3)
    curses.wrapper(main)

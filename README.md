# Liquid Analog Stream

A continuous-time neural network for raw byte streams with "Liquid" dynamics – bridging neural ODEs, SSMs, and analog hardware.

## Features

| Feature                 | Description                                                 |
| ----------------------- | ----------------------------------------------------------- |
| **Raw Byte Ingestion**  | No tokenization – streams bytes (0-255) directly            |
| **Liquid Mamba**        | Input-dependent time-step Δ = f(x) for adaptive "viscosity" |
| **NEON SIMD**           | Sub-microsecond latency (0.14µs/byte) on Apple Silicon      |
| **Register-File State** | 2KB state in Flip-Flops for nanosecond latency              |
| **Heat of Thought**     | Live visualization of neural state                          |

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Train
python3 src/train.py

# Interactive chat
python3 src/chat.py

# 🔥 Live Brain Visualization
python3 src/heat_of_thought.py

# NEON SIMD benchmark
./hardware/benchmark_neon

# Bit-accurate verification
python3 src/bit_accurate_test.py
```

## Architecture

```
Bytes → Embedding → [Liquid Mamba × N] → Output
                         │
                         └── Δ = sigmoid(f(x)) * (max - min) + min
                              └── Input-dependent time-step
```

## Project Structure

```
├── src/
│   ├── model.py              # Liquid Mamba (PyTorch)
│   ├── chat.py               # Interactive REPL
│   ├── heat_of_thought.py    # Live state visualizer ✨
│   ├── bit_accurate_test.py  # Python vs C++ verification
│   └── stateful_inference.py # O(1) per-byte
├── hardware/
│   ├── ssm_kernel_neon.cpp   # NEON SIMD kernel ✨
│   ├── ssm_kernel_fpga.cpp   # Production FPGA kernel
│   ├── ssm_kernel_sr.cpp     # Stochastic rounding
│   └── ssm_axistream.cpp     # AXI-Stream wrapper
└── neuromorphic/
    └── spiking_ssm.py        # LIF with refractory
```

## Roadmap

- [x] Phase 1-4: MVS (PyTorch)
- [x] Hardware Emulation (int8 C++)
- [x] LFSR Stochastic Rounding
- [x] Register-File FPGA Kernel
- [x] **NEON SIMD (0.14µs/byte)** ✨
- [x] **Heat of Thought Demo** ✨
- [ ] FPGA synthesis (Vivado)

## Results

| Metric            | Value           |
| ----------------- | --------------- |
| **NEON Latency**  | **0.14µs/byte** |
| Bit-Accurate MSE  | 0.00074         |
| Drift Correlation | 0.9998          |
| Saturation        | 0% at 128×16    |
| State per Layer   | 2KB             |

## License

MIT

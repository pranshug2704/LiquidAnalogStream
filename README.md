# Liquid Analog Stream

A continuous-time neural network for raw byte streams with "Liquid" dynamics – bridging neural ODEs, SSMs, and analog hardware.

## Features

| Feature                 | Description                                                 |
| ----------------------- | ----------------------------------------------------------- |
| **Raw Byte Ingestion**  | No tokenization – streams bytes (0-255) directly            |
| **Liquid Mamba**        | Input-dependent time-step Δ = f(x) for adaptive "viscosity" |
| **Stochastic Rounding** | LFSR-based SR for hardware-accurate quantization            |
| **Register-File State** | 2KB state in Flip-Flops for nanosecond latency              |
| **Fixed-Point HW**      | int8 storage, int32 accumulators for FPGA                   |

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Train
python3 src/train.py

# Interactive chat
python3 src/chat.py

# Bit-accurate verification
python3 src/bit_accurate_test.py

# Long-duration stability (use C++ for 10M+)
./hardware/testbench_sr

# Baud-rate streaming
python3 src/baud_stream.py 9600 "val: "
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
│   ├── train.py              # Training loop
│   ├── chat.py               # Interactive REPL
│   ├── bit_accurate_test.py  # Python vs C++ verification
│   ├── stateful_inference.py # O(1) per-byte
│   └── baud_stream.py        # Constant baud-rate output
├── hardware/
│   ├── ssm_kernel_fpga.cpp   # Production FPGA kernel ✨
│   ├── ssm_kernel_sr.cpp     # Stochastic rounding
│   ├── ssm_kernel_wide.cpp   # 128×16 with int32 acc
│   └── ssm_axistream.cpp     # AXI-Stream wrapper
└── neuromorphic/
    └── spiking_ssm.py        # LIF with refractory
```

## Roadmap

- [x] Phase 1-4: MVS (PyTorch)
- [x] Phase 5: MLX (Apple Silicon)
- [x] Phase 5: Hardware Emulation (int8 C++)
- [x] Liquid Refinements (bounded Δ, sub-stepping)
- [x] Stateful inference (O(1) per-byte)
- [x] LFSR Stochastic Rounding
- [x] Wide accumulators (128×16, int32)
- [x] **Register-File FPGA Kernel** ✨
- [ ] FPGA synthesis (Vivado)

## Results

| Metric            | Value                   |
| ----------------- | ----------------------- |
| Training Loss     | 0.64 (2 epochs)         |
| Stateful Latency  | **0.176ms/byte** (O(1)) |
| Bit-Accurate MSE  | **0.00074**             |
| Drift Correlation | **0.9998**              |
| Saturation        | **0%** at 128×16        |
| State per Layer   | 2KB (fits in FFs)       |

## License

MIT

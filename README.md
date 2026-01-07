# Liquid Analog Stream

A continuous-time neural network for raw byte streams with "Liquid" dynamics – bridging neural ODEs, SSMs, and analog hardware.

## Features

| Feature                  | Description                                                 |
| ------------------------ | ----------------------------------------------------------- |
| **Raw Byte Ingestion**   | No tokenization – streams bytes (0-255) directly            |
| **Liquid Mamba**         | Input-dependent time-step Δ = f(x) for adaptive "viscosity" |
| **Continuous Embedding** | Sin/cos encoding for analog-friendly representations        |
| **Multi-Scale State**    | Slow/fast state partitions for long/short-term memory       |
| **Fixed-Point HW**       | int8 C++ kernel for FPGA/analog chip mapping                |

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Train
python3 src/train.py

# Stream output at 9600 baud
python3 src/baud_stream.py 9600 "val: "

# Visualize dt (viscosity)
python3 src/visualize_dt.py

# Latency test
python3 src/latency_test.py
```

## Architecture

```
Bytes (0-255) → Embedding → [Liquid Mamba Block × N] → Output
                               │
                               └── Δ = sigmoid(f(x)) * (max - min) + min
                                   └── Input-dependent time-step
```

## Project Structure

```
├── src/
│   ├── model.py           # Liquid Mamba implementation
│   ├── train.py           # Training loop
│   ├── stream.py          # Generator-based streaming
│   ├── baud_stream.py     # Constant baud-rate output
│   └── visualize_dt.py    # dt visualization
├── hardware/
│   ├── ssm_kernel.cpp     # Q16.16 fixed-point kernel
│   └── ssm_kernel_fixed8.cpp  # int8 for analog HW
└── neuromorphic/
    └── spiking_ssm.py     # LIF neuron prototype
```

## Roadmap

- [x] Phase 1-4: MVS (PyTorch)
- [x] Phase 5: MLX (Apple Silicon)
- [x] Phase 5: Hardware Emulation (int8 C++)
- [x] Liquid Refinements (bounded Δ, sub-stepping, continuous embedding)
- [ ] Stateful inference (O(1) per-byte)
- [ ] FPGA deployment (Vivado HLS)

## Results

| Metric          | Value                       |
| --------------- | --------------------------- |
| Training Loss   | 0.64 (2 epochs)             |
| Generation      | `val: 0.999\nval: 0.999...` |
| Latency         | 4-6ms/byte                  |
| int8 Saturation | 0%                          |

## License

MIT

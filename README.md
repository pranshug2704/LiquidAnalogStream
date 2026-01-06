# Liquid Analog Stream

A multi-disciplinary project bridging high-level mathematics (Neural ODEs/SSMs) and software engineering to build a "Liquid" neural network that processes raw byte streams in continuous time.

## Project Status: Minimum Viable Stream (MVS)

We have implemented the initial "Mamba-Lite" backbone with Liquid dynamics.

### Features

- **Raw Byte Ingestion**: No tokenization. The model processes streaming raw bytes (0-255).
- **Liquid Mamba Architecture**: A pure PyTorch implementation of the Mamba State Space Model with an input-dependent time-step $\Delta$, allowing the model's "viscosity" to adapt to the data complexity.
- **Continuous-Time Logic**: Modeled as a discretized Neural ODE.

### Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the training/demo script:
   ```bash
   python3 src/train.py
   ```
   This will generate synthetic data (sine wave values), train the model, and demonstrate generation.

## Roadmap

- [x] **Phase 1**: Data Ingestion (Raw Bytes)
- [x] **Phase 2**: Mamba-Lite Backbone
- [x] **Phase 3**: Liquid Dynamics Injection
- [x] **Phase 4**: Training & Verification
- [ ] **Phase 5**: Hardware Emulation (FPGA/HLS)

## License

MIT

# Neuromorphic Integration

This directory contains code for mapping the Liquid SSM to neuromorphic hardware.

## Platforms

### Intel Loihi 2 (Lava SDK)

- Spiking neural network implementation
- Event-driven, low-power
- Install: `pip install lava-nc`

### Rain AI / Mythic

- Analog compute-in-memory
- Hardware weights as physical resistance
- Enterprise access required

### Photonic (Lightmatter)

- Optical matrix multiplication
- Speed of light inference
- Cloud API: [Lightmatter Envise](https://lightmatter.co/envise/)

## Quick Start (Lava Simulation)

```bash
pip install lava-nc
python spiking_ssm.py
```

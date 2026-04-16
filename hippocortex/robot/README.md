# `hippocortex/robot/`

Go2 Edu integration — everything that only exists because we have a physical robot.

## Planned contents

- **Sensor adapters** — RGB camera, RealSense D435i depth, L1 LiDAR, IMU; wrap raw streams into tensors consumed by `../models/` and `../data/`.
- **ROS2 nodes** — subscribe to Unitree SDK2 / CycloneDDS topics, publish model outputs back to the robot control stack.
- **Jetson runtime** — quantisation / tracing so the hybrid Mamba+Transformer model fits the 100 TOPS / 16GB envelope.
- **Task curriculum driver** — sequences the Stage-2 continual learning stream: terrain recognition → object interaction → human proximity → new environment mapping.

## Hardware reference

See [`../../docs/hardware-go2edu.md`](../../docs/hardware-go2edu.md) for the full compute/sensor budget and SDK notes.

## Not here

- Dataset classes — see `../data/`.
- Pure algorithm code — stays in `../models/` and `../cl/` so it runs unchanged in simulation.

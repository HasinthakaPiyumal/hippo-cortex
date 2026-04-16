# Hardware — Unitree Go2 Edu

Ground truth for what the Stage-2 platform can and cannot do. Update this file when we learn something new from actually running on the robot.

## Compute

| Item              | Spec                                     |
| ----------------- | ---------------------------------------- |
| Module            | NVIDIA Jetson Orin NX 16GB               |
| Peak AI perf      | 100 TOPS (INT8)                          |
| GPU memory        | 16 GB unified LPDDR5                     |
| Deployment mode   | **On-board only** — no remote base station |

Implication: the full hybrid Mamba+Transformer stack + HippoCortex consolidation must fit inside 16GB and run at real-time sensor rates. Quantisation / tracing decisions live in `hippocortex/robot/`.

## Sensors

| Sensor                  | Used for                                                              |
| ----------------------- | --------------------------------------------------------------------- |
| RGB camera              | Visual streams; each new scene = a distinct continual-learning episode. |
| Intel RealSense D435i   | Depth frames; paired with RGB for the hybrid vision stack.            |
| L1 LiDAR                | Terrain transitions (flat floor → carpet → stairs) = task boundaries. |
| IMU                     | Motion context that complements terrain recognition.                  |

## Software stack

- **ROS2** — canonical middleware.
- **Unitree SDK2 + CycloneDDS** — direct hook from our PyTorch model into the robot data pipeline, no extra abstraction layer.
- **Python SDK** on top of the above.

## Stage-2 task curriculum

The sequential task stream the Go2 Edu will learn, in order, at WSO2 Colombo (proposal, Months 8–9):

1. Terrain recognition
2. Object interaction
3. Human proximity awareness
4. New environment mapping

Each is learned in turn **without forgetting** the previous ones — this is the deployed validation of HippoCortex.

## Known risk register

_To be filled in as we hit real-world issues (sensor drift, compute thermal throttling, network outages during deployment, etc.). Keep one line per incident + a link to the meeting note where it was discussed._

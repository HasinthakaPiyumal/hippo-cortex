# stage2c_go2_deployment

Real-world deployment on the Unitree Go2 Edu at WSO2 Colombo. Inference runs entirely on the on-board Jetson Orin NX 16GB — no remote base station.

## Task stream

1. Terrain recognition (flat floor → carpet → stairs via L1 LiDAR + IMU).
2. Object interaction (via RGB + D435i depth).
3. Human proximity awareness.
4. New environment mapping.

Each is learned in turn; HippoCortex must not forget earlier tasks as new ones arrive.

## Outputs

Runs write to `../../results/stage2c_go2_deployment/<run_id>/`. Expect rosbags alongside per-task metric logs.

# stage2b_sim_mujoco

Simulator validation in Mujoco / Isaac Sim with a Go2 model before touching real hardware.

## Goal

Validate the sequential task curriculum (terrain recognition → object interaction → human proximity → new environment mapping) end-to-end in simulation. Tune for sensor-noise conditions that will match the D435i / L1 LiDAR / IMU on the real Go2 Edu.

## Outputs

Runs write to `../../results/stage2b_sim_mujoco/<run_id>/`.

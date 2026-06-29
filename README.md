# Position Based Dynamics

Implemented with Taichi

<br>

## Cloth Falling onto a Bunny

- Constraints: Distance, Bending
- Spatial Grid
- Continuous Collision Detection

| k_dist=0.2, k_bend=0.03  | k_dist=0.5, k_bend=0.03 | k_dist=0.99, k_bend=0.03 |
|:---:|:---:|:---:|
| <img width="320" height="180" alt="dist02bend003" src="https://github.com/user-attachments/assets/1da88be8-2c62-4bf7-a551-f9e9b90c619d" /> | <img width="320" height="180" alt="dist05bend003" src="https://github.com/user-attachments/assets/6e36b4a6-cc46-4063-b571-ab3c14e3da71" /> | <img width="320" height="180" alt="dist099bend003" src="https://github.com/user-attachments/assets/0ee03914-3efa-4baf-83fb-cab9b48b766e" /> |

<br>

## Bunny Passing Through a Curtain

- Constraints: Distance, Bending
- Moving Spatial Grid
- Fixed Particles
- Continuous Collision Detection

| k_dist=0.5, k_bend=0.01  | k_dist=0.5, k_bend=0.003 | k_dist=0.5, k_bend=0.1 |
|:---:|:---:|:---:|
| <img width="320" height="180" alt="dist05_bend001" src="https://github.com/user-attachments/assets/57d99a55-3be0-4cd8-a7b7-27cd4f3f69a4" /> | <img width="320" height="180" alt="dist05_bend003" src="https://github.com/user-attachments/assets/dbed5262-6cf2-4159-be6b-55d653f7334e" /> | <img width="320" height="180" alt="dist05_bend01" src="https://github.com/user-attachments/assets/2b87e1eb-22ef-4550-ac02-ec8f7e4d7a81" /> |

<br>

## Tearing

- Constraints: Distance, Volume
- Deformable Object tearing (Particle Duplication & Topology Update)
- Surface Reconstruction after tearing

| back (rubbery) | back (crumbly) |
|:---:|:---:|
| <img width="320" height="180" alt="back1back2" src="https://github.com/user-attachments/assets/3e6dc45c-431f-49f9-9b2f-a229dc2ab998" /> | <img width="320" height="180" alt="back1back2_soft" src="https://github.com/user-attachments/assets/114e996b-8718-48e8-b3b8-8e7ea59143ce" /> |

| ear | nose-butt |
|:---:|:---:|
| <img width="320" height="180" alt="ear1ear2" src="https://github.com/user-attachments/assets/f106bda7-37ac-46c7-81bc-1b9fe8b29d32" /> | <img width="320" height="180" alt="nose_butt" src="https://github.com/user-attachments/assets/044b653f-ca36-4ec0-9a2a-978aec443cdd" /> |

<br>

## Passing Between Rollers

- Constraints: Distance, Volume
- Deformable Bunny & Rigid Rollers
- Moving Surface Velocity & Friction-based dragging

| Forward Rotation | Reverse Rotation | Falling Out |
|:---:|:---:|:---:|
| <img width="320" height="180" alt="rollers_slowrotate" src="https://github.com/user-attachments/assets/7d659043-6861-493d-996f-8464b42c8232" /> | <img width="320" height="180" alt="rollers_front_reverse" src="https://github.com/user-attachments/assets/dcba7481-28aa-4976-92bb-cca84455ad8d" /> | <img width="320" height="180" alt="rollers_fall" src="https://github.com/user-attachments/assets/f1884278-aff4-4e8c-a744-2740f8f911f2" /> |

<br>

## Failures
| Too High Bending |
|:---:|
| <img width="320" height="180" alt="failure_high_bending" src="https://github.com/user-attachments/assets/780b688c-0d92-4c2c-80de-e7b6b15f16ac" /> |

<br>

## TO-DO-LIST

- ✅ friction
- ✅ restitution
- ✅ tearing
- ❌ self-collision

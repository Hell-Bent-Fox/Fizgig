# Fizgig v6.8.2

Maintenance: Krea 2 LoRAs from other trainers load again, and Klein's single-subject Identity preset moves to rank 8.

## Fixes

- **Krea 2 LoRAs from OneTrainer, AI-Toolkit and other diffusers-format trainers load again** in Repair Studio, LoRA the Explorer, LoRA Royale, Extract and as a Context LoRA. Since 6.8.0 these files were refused with "adapts nothing"; Fizgig now reads their module names as the previous Krea 2 trainer did. LoRAs trained in Fizgig were not affected.
- **Klein: the ✨ Identity single-subject preset trains at rank 8** (it was rank 4), and is renamed ✨ Identity (rank 8, single subject). Rank 8 is the lowest we recommend for a single face.

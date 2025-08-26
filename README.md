# Prophet: Fast Decoding for Diffusion Language Models

Official implementation of "Diffusion Language Models Know the Answer Before Decoding"

## Overview
Prophet is a training-free early-exit decoding paradigm for Diffusion Language Models (DLMs) that leverages the observation that correct answers often emerge early in the decoding process, well before the final step.
<img width="1065" height="480" alt="Image" src="https://github.com/user-attachments/assets/2c78909a-89bd-497c-8288-fe5539f8edb2" />

## Key Features

- **Training-free**: No additional training required, works directly with existing DLMs
- **Dynamic early-exit**: Uses confidence gap between top-2 predictions as stopping criterion
- **Significant speedup**: Up to 2.67× faster on planning tasks, 2.34× on general tasks
- **Quality preservation**: Maintains or improves generation quality compared to full decoding
- **Model-agnostic**: Compatible with different DLMs (tested on LLaDA-8B and Dream-7B)

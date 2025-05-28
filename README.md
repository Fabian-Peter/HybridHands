# HybridHands

This repository is part of a master thesis "HybridHands", focused on generating synthetic hand pose datasets. It contains scripts for parameterized hand pose generation and rendering pipeline using MANO and NIMBLE hand models.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/HybridHands.git
   cd HybridHands
   
2. Download required assets from Google Drive https://drive.google.com/drive/folders/12yMqqaYVs83P4lpWkjuobIjd--9erchv and place them in the /assets directory.

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    
## Usage Pipeline
1. Configure: Adjust pose_num variable to set number of poses, pose_std and shape_std for shape and pose noise
2. Generate Hand Poses
### For MANO model
   ```bash
   python pose_generator_MANO.py
   ```
### For NIMBLE model
   ```bash
   python pose_generator_NIMBLE.py
   ```

Output: .obj files + XYZ annotations in output/poses/

### Render Scenes
Configure: Modify paths in script if needed
   ```bash
   blenderproc run blenderproc_MANO.py
   blenderproc run blenderproc_NIMBLE.py
   ```
Output: Rendered images + JSON annotations in output/MyHAND/

### Fuse with Backgrounds
   ```bash
   python random_background_generator.py
   ```
Output: Final training images in background_images/__generated/

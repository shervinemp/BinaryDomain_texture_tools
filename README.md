# Binary Domain Texture Modding

This repository provides scripts to facilitate the modding of textures in the game **Binary Domain**. Follow the steps below to extract, edit, upscale, repack, restore, and clean up the game's texture files.

## Prerequisites

Make sure Python (>= 3.9) and the following tools are installed and added to your system `PATH`:

1. [**Crunch**](https://github.com/BinomialLLC/crunch/raw/master/bin/crunch_x64.exe)  
   Crunch is a command-line utility used for compressing and decompressing texture data. Add the directory containing `crunch_x64.exe` to your `PATH`.

2. [**NVIDIA Texture Tools Standalone App**](https://developer.nvidia.com/downloads/texture-tools-standalone-app)  
   This application is used for creating, editing, and converting texture files. Add the directory of the main executable to your `PATH`.

3. [**CubeMapGen**](https://gpuopen.com/wp-content/uploads/2017/01/cubemapgen_v1.1.exe)  
   CubeMapGen is a tool for generating and filtering cubemaps. Install and ensure the directory containing `cubemapgen_v1.1.exe` is added to your `PATH`.

4. [**ParManager**](https://github.com/Kaplas80/ParManager/releases)  
   ParManager is a utility for managing PAR archives used in SEGA-developed games like the Yakuza series and Binary Domain. It supports listing, extracting, creating, deleting, and adding files to these archives. Add the directory of the main executable to your `PATH`.

## Scripts

### Extracting Textures

Use the `extract.bat` script to extract texture files from the game's PAR archives.

1. **Navigate to the Game Directory**: Open a command prompt and navigate to the directory containing your game files. (as with all the other scripts)
2. **Run the Script**: Execute `extract.bat`:

    ```sh
    extract.bat
    ```

3. **Confirm Directory Structure**: When prompted, choose whether to create a corresponding directory structure under the staged directory.

This script processes `.par` files and extracts their contents into their respective subdirectories.

### Converting Textures

Use `python convert.py decompress "my_dds_directory"` to decompress the DDS textures into PNG.

### Editing Textures & Upscaling

Edit the extracted textures using your preferred image editing software.
Optionally, use the `python scale.py` script to upscale, sharpen and blend images together.

### Compressing Textures

Use `python convert.py compress "my_png_directory"` to compress the PNG textures into DDS.

### Editing Textures

Edit textures with the image editor of your choice.

Alternatively, upscale using the "4X-PBRify_UpscalerSPANV4" model via `python upscale.py "my_png_directory" --output_dir "my_upscaled_directory" --max_pixels 2048`.

*Note:* The max_pixels argument is set to 2048 by default since the game could crash at higher values. Use with caution.

### Repacking Textures

Repack the textures into the game's PAR archives using `python update_files.py`. (use with `--fresh` to use backup files as base)
This script pushes the staged files (under `__staged`) into the `.par` files. This operation requires the nested `.par` files created during extraction.
To avoid re-applying identical changes (from previous updates), use the `--skip` argument.

### Stashing changes

Use the `stash.bat` with the address of a file/directory under the staging directories, to move the contents into its equivalent subdirectory under `__unstaged`, and vice versa.

### Restoring Backups

This script restores the `.par` files (under `__backup`) to their original locations.

### Cleaning Up

Use `clean.bat` to clean up the extracted directories. First, read what the code does before actually using it. (might delete unintended directories)

### Testing Your Mod

Launch the game to test your newly modded textures. Ensure everything appears as expected and make any necessary adjustments.

## Note

Upscaling some textures to rather high resolutions can cause the game to get stuck on loading, possibly due to size limits or image dimensions exceeding thresholds. (further examination required)

## Future plans

1. See if [**dx-disassembler**](https://github.com/theturboturnip/dx-disassembler/) works with Binary Domain shader files.

2. ~~See if [**SonicAudioTools**](https://github.com/blueskythlikesclouds/SonicAudioTools/) works with Binary Domain csb files.~~
    At first glance, does not seem to work. [This](https://pchelpforum.net/t/some-help-with-adx-aax-csb-game-sound-files.44367/) might be interesting though.

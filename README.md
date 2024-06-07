# Binary Domain Texture Modding

This repository provides scripts to facilitate the modding of textures in the game **Binary Domain**. Follow the steps below to extract, edit, upscale, repack, restore, and clean up the game's texture files.

## Prerequisites

Make sure Python (>= 3.9) and the following tools are installed and added to your system `PATH`:

1. [**Crunch**](https://github.com/BinomialLLC/crunch/raw/master/bin/crunch_x64.exe)  
   Crunch is a command-line utility used for compressing and decompressing texture data. Add the directory containing `crunch_x64.exe` to your `PATH`.

2. [**NVIDIA Texture Tools Standalone App**](https://developer.nvidia.com/downloads/texture-tools-standalone-app)  
   This application is used for creating, editing, and converting texture files. Add the directory of the main executable to your `PATH`.

3. [**CubeMapGen**](https://gpuopen.com/wp-content/uploads/2017/01/cubemapgen_v1.1.exe)  
   CubeMapGen is a tool for generating and filtering cubemaps. Ensure the directory containing `cubemapgen_v1.1.exe` is added to your `PATH`.

4. [**ParManager**](https://github.com/Kaplas80/ParManager/releases)  
   ParManager is a utility for managing PAR archives used in SEGA-developed games like the Yakuza series and Binary Domain. It supports listing, extracting, creating, deleting, and adding files to these archives. Add the directory of the main executable to your `PATH`.

## Extracting Textures

Use the `extract.bat` script to extract texture files from the game's PAR archives.

### Usage

1. **Navigate to the Game Directory**: Open a command prompt and navigate to the directory containing your game files.
2. **Run the Script**: Execute `extract.bat`:

    ```sh
    extract.bat
    ```

3. **Confirm Directory Structure**: When prompted, choose whether to create a corresponding directory structure under the modified directory.

This script processes `.par` files and extracts their contents into their respective subdirectories under the modified directory (`__modified`).

## Converting Textures

Use `python convert.py decompress` to decompress the DDS files into PNG.

## Editing Textures & Upscaling

Edit the extracted textures using your preferred image editing software.
Optionally, use the `python scale.py` script to upscale, sharpen and blend images together.

## Compressing Textures

Use `python convert.py compress` to compress the PNG files into DDS.

## Repacking Textures

After editing, repack the textures into the game's PAR archives using the `update_files.py` script.

### Usage

1. **Navigate to the Game Directory**: Open a command prompt and navigate to the directory containing your game files.
2. **Run the Script**: Execute `update_files.py`:

    ```sh
    python update_files.py
    ```

This script pushes the modified files (under `__modified`) into the `.par` files. This operation requires the nested `.par` files created during extraction.

## Restoring Backups

Use the `restore.bat` script to restore original `.par` files from the backup directory (`__backup`).

### Usage

1. **Navigate to the Game Directory**: Open a command prompt and navigate to the directory containing your game files.
2. **Run the Script**: Execute `restore.bat`:

    ```sh
    restore.bat
    ```

3. **Confirm Restoration**: Choose whether to restore all backups immediately or one by one.
4. **Delete Backup Directory**: After restoring, confirm whether to delete the backup directory.

This script restores the `.par` files (under `__backup`) to their original locations.

## Cleaning Up

Use the `clean.bat` script to clean up the extracted directories.

### Usage

1. **Navigate to the Game Directory**: Open a command prompt and navigate to the directory containing your game files.
2. **Run the Script**: Execute `clean.bat`:

    ```sh
    clean.bat
    ```

3. **Confirm Removal**: When prompted, choose whether to remove the extracted directories.

The script removes all directories with the suffix `.par_` except the modified directory.

## Testing Your Mod

Launch the game to test your newly modded textures. Ensure everything appears as expected and make any necessary adjustments.

## Note

Upscaling some textures, especially those related to the main menu, may cause the game to crash. Proceed with caution when modifying these textures.

## Future plans

1. See if [**dx-disassembler**](https://github.com/theturboturnip/dx-disassembler/) works with Binary Domain shader files.

2. See if [**SonicAudioTools**](https://github.com/blueskythlikesclouds/SonicAudioTools/) works with Binary Domain csb files.
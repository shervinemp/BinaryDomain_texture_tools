# Binary Domain Texture Modding

This repository provides scripts to facilitate the modding of textures in the game **Binary Domain**. Follow the steps below to extract, edit, upscale, repack, restore, and clean up the game's texture files.

## Prerequisites

Ensure the following tools are installed and added to your system `PATH`:

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

This script processes `.par` files and extracts the contents into respective directories.

## Editing Textures

Edit the extracted textures using your preferred image editing software.

## Upscaling and Compressing Textures

Use the `scale.py` script to upscale and compress textures. The script can be configured to perform either or both functions based on the provided arguments.

### Usage

```sh
python scale.py source_dir output_dir [--prescaled_dir prescaled_dir] [--temp_dir temp_dir] [--interpolation interpolation] [--scale scale] [--blend blend] [--sharpen sharpen] [--silent] [--recurse] [--skip] [--keep] [--no_compress] [-p processes]
```

### Command-Line Arguments

- `source_dir`: Directory containing input DDS files to be processed.
- `output_dir`: Directory where the processed files will be written.
- `--prescaled_dir`: Directory for pre-scaled images to blend with the upscaled images. To perform only compression, set this to a valid directory and use `--blend 1`.
- `--temp_dir`: Directory for temporary files during processing (default: `.tmp`). This helps manage intermediate steps and ensures smooth operation.
- `--interpolation`: Type of interpolation for upscaling (e.g., nearest, bilinear, bicubic) (default: `auto`). This determines the quality and method of upscaling.
- `--scale`: Scale factor for upscaling (default: `2.0`). To avoid upscaling, set this to `1`.
- `--blend`: Blending percentage for pre-scaled images (default: `0.5`). To perform only compression, set this to `1`.
- `--sharpen`: Type of sharpening filter (e.g., unsharp_mask, CAS) to enhance image details.
- `--silent`: Silences warnings and outputs for a cleaner execution.
- `--recurse`: Recurse through input directories to process all nested files.
- `--skip`: Skips overwriting temporary files, useful for resuming interrupted processes.
- `--keep`: Keeps temporary files after processing, useful for debugging or additional processing steps.
- `--no_compress`: Skips the compression step, useful if only upscaling is needed.
- `-p`: Number of parallel processes to use (default: 1), which can speed up the processing time.

By adjusting these options, you can flexibly and efficiently handle texture files to meet your specific requirements, whether you need upscaling, compression, or both.

## Repacking Textures

After editing, repack the textures into the game's PAR archives using the `update_files.py` script.

### Usage

1. **Navigate to the Repository Directory**: Open a command prompt and navigate to the directory containing your repository.
2. **Run the Script**: Execute `update_files.py`:

    ```sh
    python update_files.py
    ```

This script copies the modified files to a temporary directory, updates the `.par` files, and overwrites the game files.

## Restoring Backups

Use the `restore.bat` script to restore original texture files from backups.

### Usage

1. **Navigate to the Backup Directory**: Open a command prompt and navigate to the directory containing your backup files.
2. **Run the Script**: Execute `restore.bat`:

    ```sh
    restore.bat
    ```

3. **Confirm Restoration**: Choose whether to restore all backups or one at a time.
4. **Delete Backup Directory**: After restoring, confirm whether to delete the backup directory.

This script restores the `.par` files to their original locations.

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

## Important Note

Upscaling some textures, especially those related to the main opening menu, may cause the game to crash. Proceed with caution when modifying these textures.

## Conclusion

By following these instructions, you can customize the visual elements of **Binary Domain** to your liking. Happy modding!
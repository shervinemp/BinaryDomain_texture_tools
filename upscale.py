import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

try:
    from spandrel import ImageModelDescriptor, ModelLoader
except ImportError as e:
    print(f"Error importing spandrel: {e}")
    print("Please install spandrel (pip install spandrel).")
    sys.exit(1)


def collect_input_files(input_path, recursive=True):
    """Collect all PNG files from a directory."""

    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        return []

    if input_path.is_dir():
        # Collect PNG files from the directory
        glob_pattern = "**/*.png" if recursive else "*.png"
        found_files = list(input_path.glob(glob_pattern))
        print(f"Found {len(found_files)} .png files in {input_path}")
        return found_files
    elif input_path.is_file() and input_path.suffix == ".png":
        # Process individual file
        return [input_path]

    return []


def load_image(file_path):
    """Load an image and handle alpha channel separately if needed."""
    img = Image.open(file_path)
    img_rgba = img.convert("RGBA")
    rgb = img_rgba.split()[:3]
    alpha = img_rgba.split()[3:4]
    has_alpha = "A" in img.mode or (img.mode == "RGBA")

    rgb = torch.from_numpy(np.array(rgb)).float()
    alpha = (
        torch.from_numpy(np.array(alpha)).repeat((3, 1, 1)).float()
        if has_alpha
        else None
    )

    return rgb, alpha


def load_model(
    model_path=None, device="cuda"
):  # Added model_path parameter with default value
    """Load the model from the specified path."""
    if model_path is None:
        model_path = Path(__file__).parent / "4x-PBRify_UpscalerSPANV4.pth"
    print(f"Loading model from {model_path}...")
    model = ModelLoader().load_from_file(model_path)

    if not isinstance(model, ImageModelDescriptor):
        raise ValueError("The loaded model is not an image-to-image model.")

    # Set device and evaluate mode
    model = model.to(device).eval()
    return model


def apply_max_pixels(img, max_pixels=2048):
    """Resize image if any dimension exceeds max_pixels while preserving aspect ratio."""
    width, height = img.size
    needs_resize = width > max_pixels or height > max_pixels

    if not needs_resize:
        return img

    aspect_ratio = width / height
    if aspect_ratio > 1:  # Width is larger
        new_width = min(width, max_pixels)
        new_height = new_width / aspect_ratio
    else:  # Height is larger
        new_height = min(height, max_pixels)
        new_width = new_height * aspect_ratio

    return img.resize((int(new_width), int(new_height)), Image.LANCZOS)


def run_inference(input_path: Path, output_dir: Path, recursive=True, max_pixels=2048):
    """Load a model and run inference on input tensors."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory set to: {output_dir}")

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = load_model(device=device)

    # Collect input files
    input_files = collect_input_files(input_path, recursive)
    if not input_files:
        print("No input files found to process.")
        return

    # Process in batches
    for file_path in input_files:
        try:
            rgb, alpha = load_image(file_path)
            t_ = [rgb]
            if alpha is not None:
                t_.append(alpha)

            input_batch = torch.stack(t_, dim=0).to(device)

            with torch.no_grad():
                output = model(input_batch)
                print(f"Inference completed on batch, output shape: {output.shape}")

            out, *out_alpha_ = (
                output.cpu().numpy().transpose(0, 2, 3, 1).astype("uint8")
            )
            if len(out_alpha_):
                out_alpha = out_alpha_[0][..., :1]
                out = np.concat((out, out_alpha), axis=2)

            img = Image.fromarray(out)
            img = apply_max_pixels(img, max_pixels)

            rel_path = file_path.relative_to(input_path)
            output_path = output_dir / rel_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            # Save the image
            img.save(output_path)
        except Exception as e:
            print(f"Error during inference: {e}")
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load a model and run inference on input tensors.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input_path", type=Path, help="Path to the input directory or PNG file."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/inference_results"),
        help="Directory to save output images.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=True,
        help="Recursively search directories for PNG files.",
    )
    parser.add_argument(
        "--max_pixels",
        type=int,
        default=2048,
        help="Maximum pixels for the largest dimension of an image. Images exceeding this resolution will be downscaled to fit within this limit while preserving aspect ratio.",
    )

    args = parser.parse_args()
    print(f"Command-line arguments parsed: {vars(args)}")

    run_inference(
        input_path=args.input_path,
        output_dir=args.output_dir,
        recursive=args.recursive,
        max_pixels=args.max_pixels,
    )

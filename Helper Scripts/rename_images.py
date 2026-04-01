import argparse
from pathlib import Path


def rename_images(folder_path: str | Path) -> None:
    """Rename PNG files in a folder sequentially (001, 002, 003, etc.)"""

    folder = Path(folder_path)

    if not folder.is_dir():
        raise ValueError(f"{folder} is not a valid directory")

    png_files = sorted(folder.glob("*.png"))

    if not png_files:
        print("No PNG files found in the folder")
        return

    print(f"Found {len(png_files)} PNG files. Starting rename...")

    for index, file_path in enumerate(png_files, start=1):
        new_name = f"{index:03d}.png"
        new_path = folder / new_name

        file_path.rename(new_path)
        print(f"Renamed: {file_path.name} -> {new_name}")

    print("Done!")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rename PNG files to 001.png, 002.png, ...")
    parser.add_argument("--folder", default=None, help="Folder path containing PNG files")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    folder_path = args.folder or input("Enter the folder path: ").strip()
    rename_images(folder_path)

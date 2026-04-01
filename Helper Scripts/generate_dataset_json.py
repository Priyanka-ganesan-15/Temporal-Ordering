import argparse
import json
from pathlib import Path


def generate_dataset_json(data_dir: str | Path, output_path: str | Path) -> None:
    """Generate a sequence manifest from Data/* folders."""
    data_folder = Path(data_dir)
    output_file = Path(output_path)

    if not data_folder.is_dir():
        raise ValueError(f"{data_folder} is not a valid directory")

    sequences = sorted([folder for folder in data_folder.iterdir() if folder.is_dir()])
    if not sequences:
        raise ValueError("No subfolders found in the data directory")

    dataset = []
    for seq_folder in sequences:
        png_files = sorted(seq_folder.glob("*.png"))
        if not png_files:
            print(f"Skipping {seq_folder.name}: no PNG files found")
            continue

        entry = {
            "sequence_id": seq_folder.name.lower(),
            "category": "",
            "caption": "",
            "num_frames": len(png_files),
            "frames": [
                str(file_path.relative_to(data_folder.parent)).replace("\\", "/")
                for file_path in png_files
            ],
        }
        dataset.append(entry)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as file_obj:
        json.dump(dataset, file_obj, indent=2)

    print(f"Generated JSON with {len(dataset)} sequences -> {output_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate dataset manifest JSON")
    parser.add_argument("--data-dir", default=None, help="Path to Data folder")
    parser.add_argument("--output-path", default=None, help="Output JSON path")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data_dir = args.data_dir or input("Enter path to the Data folder: ").strip()
    output_path = args.output_path or input(
        "Enter output JSON file path (e.g. dataset.json): "
    ).strip()
    generate_dataset_json(data_dir, output_path)

from argparse import ArgumentParser
import os
from pathlib import Path

from tqdm import tqdm


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("dest", type=Path)
    args = parser.parse_args()

    dest: Path = args.dest

    for file in tqdm(
        list(
            sorted(
                args.source.glob("*.zip"),
                key=lambda p: int(p.stem.split("-")[0]),
            )
        )
):
        file: Path
        book: str = file.stem.split("-")[0]
        try:
            target = dest / "/".join(list(book))
            
            # Create nested directories if they don't exist
            target.mkdir(parents=True, exist_ok=True)

            # Create a hardlink from the source to the nested destination
            # e.g. {source}/123.zip -> {dest}/1/2/3/123.zip
            os.link(file, target / file.name)
        except:
            print(f"Failed to link {file} -> {book}")
            raise
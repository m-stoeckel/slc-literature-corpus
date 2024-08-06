from pathlib import Path
import requests
from tqdm import tqdm


def fetch_ls_r() -> list[str]:
    response = requests.get(url="https://www.gutenberg.org/dirs/ls-R")
    response.raise_for_status()
    ls_r = response.text
    print(
        f"fetched {len(ls_r)} ls-R output lines containing {sum(1 if l.strip() and not l.endswith(':') else 0 for l in ls_r)} file references"
    )
    return ls_r.splitlines()


def parse_ls_r(ls_r: list[str]) -> dict[str, Path]:
    last_root = Path(".")
    files = {}
    for line in tqdm(ls_r):
        line = line.strip()
        if line.startswith(".") and line.endswith(":"):
            last_root = Path(line.removesuffix(":"))
        elif line.endswith(".zip"):
            file_path = last_root / line

            # Skip all old versions
            if "old" in str(file_path):
                continue

            # Exhaustive case matching all Gutenberg text files
            match file_path.stem.split("-"):
                ## ls -R output sorts -0 before all others (-0 is UTF-8 encoded text)
                ## Others are only fallbacks in case -0 is not present
                ## Last fallback match is [book] without any suffixes
                case [book, ("0" | "8" | "utf8")] | [
                    book
                ] if book.isdigit() and not book.startswith("0"):
                    if book in files:
                        continue

                    files[book] = file_path
                case _:
                    continue
    print(f"parsed {len(files)} files from ls -R output")
    return files


def obtain_urls(write: str = "data/urls.txt") -> dict[str, Path]:
    ls_r = fetch_ls_r()
    files = parse_ls_r(ls_r)
    if write:
        with open(write, "w") as f:
            f.write(
                "\n".join(
                    f"https://www.gutenberg.org/files/{book}/{file.name}"
                    for book, file in files.items()
                )
            )
    return files

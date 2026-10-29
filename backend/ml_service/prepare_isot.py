import hashlib
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
DATASET_URL = (
    "https://huggingface.co/datasets/"
    "Phoenyx83/ISOT-Fake-News-Dataset-FineTuned-2022/resolve/main/"
)
SOURCES = {
    "fake_news_isot_fake.csv": {
        "path": "raw/Fake%20ok.csv?download=true",
        "sha256": "3974cb40f8d15590ea88679dddbff00dd3c8ac46281156db72f84bc57adb8151",
    },
    "fake_news_isot_real.csv": {
        "path": "raw/True%20ok%20v2.csv?download=true",
        "sha256": "b81e8ee595f0a7d79e52caca7b44468fd79279f05f46bfda0fb656f13e1b2de8",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_source(filename: str, source: dict[str, str]) -> Path:
    target = DATA_DIR / filename
    if target.exists():
        if sha256_file(target) != source["sha256"]:
            raise ValueError(f"Existing dataset file failed checksum verification: {target}")
        return target

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    try:
        with urlopen(DATASET_URL + source["path"], timeout=60) as response:
            with temporary.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
        if sha256_file(temporary) != source["sha256"]:
            raise ValueError(f"Downloaded dataset file failed checksum verification: {filename}")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def prepare() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    fake_path = download_source("fake_news_isot_fake.csv", SOURCES["fake_news_isot_fake.csv"])
    real_path = download_source("fake_news_isot_real.csv", SOURCES["fake_news_isot_real.csv"])

    fake = pd.read_csv(fake_path, encoding="utf-8", on_bad_lines="error")
    real = pd.read_csv(real_path, encoding="utf-8", on_bad_lines="error")
    for name, records, expected_label in (
        ("fake", fake, 1),
        ("real", real, 0),
    ):
        if not {"type", "text"}.issubset(records.columns):
            raise ValueError(f"The ISOT {name} CSV must include type and text columns.")
        labels = set(records["type"].dropna().unique())
        if labels != {expected_label}:
            raise ValueError(
                f"Unexpected labels in ISOT {name} CSV: {sorted(labels)}"
            )

    data = pd.concat(
        [
            fake[["text", "type"]].rename(columns={"type": "label"}),
            real[["text", "type"]].rename(columns={"type": "label"}),
        ],
        ignore_index=True,
    )
    data["label"] = data["label"].map({0: "REAL", 1: "FAKE"})
    output = DATA_DIR / "isot_fake_news.csv"
    data.dropna(subset=["text", "label"]).to_csv(output, index=False, encoding="utf-8")
    print(
        f"Prepared {len(data.dropna(subset=['text', 'label'])):,} labeled articles "
        f"at {output}"
    )
    return output


if __name__ == "__main__":
    prepare()

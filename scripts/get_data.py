"""Download Fashion-MNIST and convert it to the flat binaries read by src/core/data.cpp.

Output (in --out):
    train_images.f32, test_images.f32   n x 784 float32, pixels / 255
    train_labels.u8,  test_labels.u8    n uint8
    manifest.json                       source URLs and SHA-256 of every file

Standard library only, so it also runs on Atlas/Vanda login nodes.
Source: the dataset authors' download host (Xiao, Rasul & Vollgraf 2017), checked
against the published MD5 values; also mirrored on Hugging Face.
"""
import argparse
import array
import gzip
import hashlib
import json
import pathlib
import shutil
import ssl
import struct
import urllib.request

# The authors' download host (linked from github.com/zalando-research/fashionmnist);
# MD5 values are the ones published there and used by torchvision.
BASE = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
FILES = {
    "train_images": ("train-images-idx3-ubyte.gz", "8d4fb7e6c68d591d4c3dfef9ec88bf0d"),
    "train_labels": ("train-labels-idx1-ubyte.gz", "25c81989df183df01b3e8a0aad5dffbe"),
    "test_images": ("t10k-images-idx3-ubyte.gz", "bef4ecab320f06d8554ea6380940ec79"),
    "test_labels": ("t10k-labels-idx1-ubyte.gz", "bb300cfdad3c16e7a12a480ee83cd310"),
}


def ssl_context():
    """Some Python builds (e.g. miniconda) ship without system CA certificates."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def download(url, target):
    with urllib.request.urlopen(url, context=ssl_context()) as response, open(target, "wb") as f:
        shutil.copyfileobj(response, f)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_idx(raw):
    """Parse an IDX file: returns (dimensions, payload bytes)."""
    _, dtype, ndim = struct.unpack(">HBB", raw[:4])
    if dtype != 0x08:
        raise ValueError("expected unsigned byte IDX data")
    dims = list(struct.unpack(">" + "I" * ndim, raw[4 : 4 + 4 * ndim]))
    return dims, raw[4 + 4 * ndim :]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="data/fashion_mnist")
    args = parser.parse_args()
    out = pathlib.Path(args.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)

    manifest = {"source": BASE, "files": {}}
    for key, (name, md5) in FILES.items():
        gz_path = out / "raw" / name
        if not gz_path.exists():
            print(f"downloading {name}")
            download(BASE + name, gz_path)
        if hashlib.md5(gz_path.read_bytes()).hexdigest() != md5:
            gz_path.unlink()
            raise SystemExit(f"MD5 mismatch for {name}; file removed, rerun")
        dims, payload = read_idx(gzip.decompress(gz_path.read_bytes()))
        split, kind = key.split("_")
        if kind == "images":
            n, rows, cols = dims
            assert rows * cols == 784, dims
            floats = array.array("f", (b / 255.0 for b in payload))
            target = out / f"{split}_images.f32"
            target.write_bytes(floats.tobytes())
        else:
            (n,) = dims
            target = out / f"{split}_labels.u8"
            target.write_bytes(payload)
        manifest["files"][name] = {"sha256": sha256(gz_path), "samples": n}
        manifest["files"][target.name] = {"sha256": sha256(target), "samples": n}
        print(f"{target}: {n} samples")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()

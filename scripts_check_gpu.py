import sys

import torch


def main() -> int:
    print("Python:", sys.version.split()[0])
    print("PyTorch:", torch.__version__)
    print("CUDA dostępna:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        print("Brak CUDA. Sprawdź instalację PyTorch i sterownik NVIDIA.")
        return 1

    properties = torch.cuda.get_device_properties(0)
    print("CUDA w PyTorch:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM:", round(properties.total_memory / 1024**3, 2), "GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

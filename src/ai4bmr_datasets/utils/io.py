from pathlib import Path

import numpy as np
from PIL import Image

# Default lossless compression configurations
DEFAULT_ZARR_CODECS = [
    {'name': 'bytes', 'configuration': {'endian': 'little'}},  # ArrayBytesCodec
    {'name': 'blosc', 'configuration': {'cname': 'zstd', 'clevel': 1, 'shuffle': 'shuffle'}}  # BytesBytesCodec
]
DEFAULT_TIFF_COMPRESSION = "zstd"


def imread(img_path: Path | str, engine: str | None = None) -> np.ndarray:
    """
    Read an image from various formats.

    Args:
        img_path: Path to the image file
        engine: File format override (auto-detected from extension if None)

    Returns:
        Image as numpy array
    """
    img_path = Path(img_path)
    engine = engine or img_path.suffix.lower()

    match engine:
        case '.tiff' | '.tif':
            import tifffile
            return tifffile.imread(img_path)
        case '.zip':
            import zarr
            store = zarr.ZipStore(str(img_path), mode="r")
            try:
                arr = zarr.open_array(store=store, mode='r')
                return np.asarray(arr)
            finally:
                store.close()
        case '.zarr':
            import zarr
            arr = zarr.open_array(str(img_path), mode='r')
            return np.asarray(arr)
        case _:
            raise ValueError(f'Unsupported image format {engine} for {img_path}')


def imsave(
        img: np.ndarray,
        save_path: Path | str,
        engine: str | None = None,
        **kwargs
) -> None:
    """
    Save a CHW image/mask array to various formats.

    Args:
        img: Image array to save (typically in CHW format)
        save_path: Output file path
        engine: File format override (auto-detected from extension if None)
        **kwargs: Format-specific arguments:
            - For zarr: chunks, codecs, zarr_format, etc.
            - For tiff: compression, tile, etc.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    engine = engine or save_path.suffix.lower()

    match engine:
        case '.zarr':
            import zarr
            # Use provided codecs or default to lossless compression
            codecs = kwargs.pop('codecs', DEFAULT_ZARR_CODECS)
            chunks = kwargs.pop('chunks', None)
            zarr_format = kwargs.pop('zarr_format', 3)

            z = zarr.open_array(
                store=str(save_path),
                mode='w',
                shape=img.shape,
                dtype=img.dtype,
                chunks=chunks,
                zarr_format=zarr_format,
                codecs=codecs,
                **kwargs
            )
            z[...] = img

        case '.zip':
            import zarr
            # Use provided codecs or default to lossless compression
            codecs = kwargs.pop('codecs', DEFAULT_ZARR_CODECS)
            chunks = kwargs.pop('chunks', None)
            zarr_format = kwargs.pop('zarr_format', 3)

            store = zarr.ZipStore(str(save_path), mode="w")
            try:
                z = zarr.open_array(
                    store=store,
                    mode='w',
                    shape=img.shape,
                    dtype=img.dtype,
                    chunks=chunks,
                    zarr_format=zarr_format,
                    codecs=codecs,
                    **kwargs
                )
                z[...] = img
            finally:
                store.close()

        case '.tiff' | '.tif':
            import tifffile
            # Use provided compression or default to lossless compression
            compression = kwargs.pop('compression', DEFAULT_TIFF_COMPRESSION)

            tifffile.imwrite(
                save_path,
                img,
                compression=compression,
                **kwargs
            )

        case _:
            raise ValueError(f'Unsupported image format {engine} for {save_path}')


def read_region(
        img_path: Path | str,
        y: int,
        x: int,
        height: int,
        width: int,
        channel: int | slice = slice(None),
        level: int = 0,
        engine: str | None = None
) -> np.ndarray | Image.Image:
    """
    Read a region from an image file.

    Args:
        img_path: Path to the image file
        y: Y coordinate (top)
        x: X coordinate (left)
        height: Height of region
        width: Width of region
        channel: Channel selection (int, slice, or slice(None) for all)
        level: Pyramid level for multi-resolution formats
        engine: File format override

    Returns:
        Image region as numpy array or PIL Image
    """
    img_path = Path(img_path)
    engine = engine or img_path.suffix.lower()

    match engine:
        case '.zarr':
            import zarr
            z = zarr.open_array(str(img_path), mode='r')
            return np.asarray(z[channel, y:y + height, x:x + width])

        case '.zip':
            import zarr
            store = zarr.ZipStore(str(img_path), mode="r")
            try:
                z = zarr.open_array(store=store, mode='r')
                return np.asarray(z[channel, y:y + height, x:x + width])
            finally:
                store.close()

        case '.tiff' | '.tif':
            import openslide
            slide = openslide.open_slide(str(img_path))
            img = slide.read_region(
                location=(x, y),
                level=level,
                size=(width, height)
            )
            return img.convert("RGB")

        case _:
            raise ValueError(f'Unsupported image format {engine} for {img_path}')


def save_mask(
        mask: np.ndarray,
        save_path: Path | str,
        compression: str | None = "deflate"
) -> None:
    """
    Save a mask array as a TIFF file with appropriate dtype.

    Args:
        mask: Mask array
        save_path: Output file path
        compression: TIFF compression method
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    import tifffile

    # Choose appropriate dtype based on max value
    max_val = mask.max()
    if max_val < 256:
        mask = mask.astype(np.uint8)
    elif max_val < 65536:
        mask = mask.astype(np.uint16)
    else:
        mask = mask.astype(np.uint32)

    tifffile.imwrite(save_path, mask, compression=compression)
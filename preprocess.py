"""
  Step 1. Otsu Thresholding on low-res thumbnail -> Tissue Mask
  Step 2.  Smart Patching at highest resolution using the tissue mask
  Step 3.  Macenko Stain Normalization patch-by-patch
"""

from __future__ import annotations

import gc
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, Tuple

import cv2
import numpy as np
from PIL import Image
from scipy import linalg
from tqdm import tqdm

warnings.filterwarnings("ignore")


def _bootstrap_openslide() -> bool:
    """
    Try to import openslide_bin first so it can register the bundled
    native library. Falls back gracefully if the package is absent.
    Returns True if openslide is usable.
    """
    try:
        import openslide_bin
    except ImportError:
        print(
            "openslide_bin not found."
        )

    try:
        import openslide
        return True
    except ImportError:
        print("openslide-python not installed.")
        return False


_OPENSLIDE_OK = _bootstrap_openslide()


class SlideReader(Protocol):
    @property
    def level_count(self) -> int: ...
    @property
    def level_dimensions(self) -> list[tuple[int, int]]: ...
    @property
    def level_downsamples(self) -> list[float]: ...
    def get_thumbnail(self, size: tuple[int, int]) -> Image.Image: ...
    def read_region(self, location: tuple[int, int], level: int,
                    size: tuple[int, int]) -> Image.Image: ...
    def close(self) -> None: ...
    @property
    def properties(self) -> dict: ...


class OpenSlideReader:
    def __init__(self, path: str):
        import openslide
        self._slide = openslide.OpenSlide(path)

    @property
    def level_count(self):
        return self._slide.level_count

    @property
    def level_dimensions(self):
        return self._slide.level_dimensions

    @property
    def level_downsamples(self):
        return self._slide.level_downsamples

    @property
    def properties(self):
        return self._slide.properties

    def get_thumbnail(self, size: tuple[int, int]) -> Image.Image:
        return self._slide.get_thumbnail(size)

    def read_region(self, location: tuple[int, int], level: int,
                    size: tuple[int, int]) -> Image.Image:
        return self._slide.read_region(location, level, size)

    def close(self):
        self._slide.close()


class TifffileReader:
    def __init__(self, path: str):
        import tifffile
        self._tif = tifffile.TiffFile(path)
        self._path = path
        # Build pyramid: only keep pages with 2-D RGB data
        self._pages = [
            p for p in self._tif.pages
            if len(p.shape) == 3 and p.shape[2] in (3, 4)
        ]
        if not self._pages:
            raise ValueError("tifffile: no RGB pages found in the WSI.")

        # Sort by area descending (level 0 first)
        self._pages.sort(key=lambda p: p.shape[0] * p.shape[1], reverse=True)
        print(
            "tifffile: found %d pyramid level(s): %s",
            len(self._pages),
            [f"{p.shape[1]}x{p.shape[0]}" for p in self._pages],
        )

        # Pre-compute downsamples relative to level 0
        W0, H0 = self._pages[0].shape[1], self._pages[0].shape[0]
        self._dims   = [(p.shape[1], p.shape[0]) for p in self._pages]
        self._dsamp  = [W0 / d[0] for d in self._dims]

    @property
    def level_count(self):
        return len(self._pages)

    @property
    def level_dimensions(self):
        return self._dims

    @property
    def level_downsamples(self):
        return self._dsamp

    @property
    def properties(self):
        # Return a minimal properties dict
        return {}

    def get_thumbnail(self, size: tuple[int, int]) -> Image.Image:
        """Return the lowest-resolution level resized to `size`."""
        small = self._pages[-1].asarray()
        if small.shape[2] == 4:
            small = small[:, :, :3]
        img = Image.fromarray(small, "RGB")
        img.thumbnail(size, Image.LANCZOS)
        return img

    def read_region(self, location: tuple[int, int], level: int,
                    size: tuple[int, int]) -> Image.Image:
        """
        Read a (size[0] × size[1]) patch at `level` starting at `location`
        (location is given in level-0 pixel coordinates).
        """
        level = min(level, self.level_count - 1)
        ds = self._dsamp[level]
        # Convert level-0 location → this level's coordinates
        x_lv = int(location[0] / ds)
        y_lv = int(location[1] / ds)
        W_lv, H_lv = self._dims[level]
        pw, ph = size  # requested patch size in this level's pixels

        # Clamp to page bounds
        x1 = min(x_lv, W_lv)
        y1 = min(y_lv, H_lv)
        x2 = min(x_lv + pw, W_lv)
        y2 = min(y_lv + ph, H_lv)

        # tifffile can read sub-regions efficiently when the file is tiled
        try:
            region = self._pages[level].asarray()[y1:y2, x1:x2]
        except Exception:
            # Last resort: read full page (slow for large levels)
            full = self._pages[level].asarray()
            region = full[y1:y2, x1:x2]

        if region.shape[2] == 4:
            region = region[:, :, :3]

        img = Image.fromarray(region, "RGB")
        # Pad to requested size if we hit the image boundary
        if img.size != (pw, ph):
            padded = Image.new("RGB", (pw, ph), (255, 255, 255))
            padded.paste(img, (0, 0))
            img = padded
        return img

    def close(self):
        self._tif.close()


def open_slide(path: str) -> SlideReader:
    """
    Try OpenSlide first; fall back to tifffile if it fails.
    Raises RuntimeError if neither backend can open the file.
    """
    if _OPENSLIDE_OK:
        try:
            reader = OpenSlideReader(path)
            return reader
        except Exception as e:
            print("OpenSlide failed (%s)", e)

    try:
        import tifffile
        reader = TifffileReader(path)
        return reader
    except ImportError:
        raise RuntimeError(
            "Neither openslide nor tifffile can open the file.\n"
            "Run: pip install openslide-bin openslide-python tifffile --upgrade"
        )
    except Exception as e:
        raise RuntimeError(f"tifffile also failed: {e}")


@dataclass
class PipelineConfig:
    wsi_path: str = "slide.svs"
    output_dir: str = "output_patches"

    mask_level: int = 2
    otsu_channel: str = "saturation"   # "gray" | "saturation"
    tissue_threshold: float = 0.5

    patch_level: int = 0
    patch_size: int = 256
    stride: int = 256

    normalize: bool = True
    reference_patch_path: Optional[str] = None
    macenko_percentile: float = 99.0
    macenko_beta: float = 0.15

    save_mask: bool = True
    save_format: str = "png"
    max_patches: Optional[int] = None


class TissueMasker:

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg

    def build_mask(self, slide: SlideReader) -> tuple[np.ndarray, tuple[int, int]]:
        level = min(self.cfg.mask_level, slide.level_count - 1)
        thumb_size = slide.level_dimensions[level]   # (W, H)

        print("Thumbnail level %d → %s px", level, thumb_size)
        thumbnail = slide.get_thumbnail(thumb_size)
        thumb_np = np.array(thumbnail.convert("RGB"))   # (H, W, 3)

        if self.cfg.otsu_channel == "saturation":
            hsv = cv2.cvtColor(thumb_np, cv2.COLOR_RGB2HSV)
            channel = hsv[:, :, 1]
            _, mask = cv2.threshold(channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            gray = cv2.cvtColor(thumb_np, cv2.COLOR_RGB2GRAY)
            _, mask = cv2.threshold(
                255 - gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)

        print("Tissue coverage: %.1f %%", mask.mean() / 255 * 100)
        return mask, thumb_size

    def save_mask(self, mask: np.ndarray, out_dir: Path) -> None:
        p = out_dir / "tissue_mask.png"
        Image.fromarray(mask).save(str(p))
        print("Tissue mask → %s", p)


class MacenkoNormalizer:

    def __init__(self, beta: float = 0.15, percentile: float = 99.0):
        self.beta = beta
        self.percentile = percentile
        self._stain_matrix_ref: Optional[np.ndarray] = None
        self._max_conc_ref: Optional[np.ndarray] = None

    @staticmethod
    def _rgb_to_od(img: np.ndarray) -> np.ndarray:
        return -np.log(np.maximum(img.astype(np.float64), 1) / 255.0)

    @staticmethod
    def _od_to_rgb(od: np.ndarray) -> np.ndarray:
        return np.clip(np.exp(-od) * 255, 0, 255).astype(np.uint8)

    def _get_stain_matrix(
        self, img_rgb: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        od = self._rgb_to_od(img_rgb).reshape(-1, 3)
        od = od[np.linalg.norm(od, axis=1) > self.beta]

        # Fallback for near-blank patches
        if od.shape[0] < 10:
            return (
                np.array([[0.6500, 0.7042, 0.2860],
                          [0.0704, 0.9911, 0.1120]]),
                np.array([1.0, 1.0]),
            )

        _, _, Vt = linalg.svd(od, full_matrices=False)
        plane = Vt[:2].T              # (3, 2)
        proj = od @ plane             # (N, 2)
        angles = np.arctan2(proj[:, 1], proj[:, 0])

        phi_min = np.percentile(angles, 100 - self.percentile)
        phi_max = np.percentile(angles, self.percentile)

        v1 = plane @ np.array([np.cos(phi_min), np.sin(phi_min)])
        v2 = plane @ np.array([np.cos(phi_max), np.sin(phi_max)])
        v1 /= np.linalg.norm(v1) + 1e-6
        v2 /= np.linalg.norm(v2) + 1e-6

        if v1[0] < v2[0]:
            v1, v2 = v2, v1
        stain = np.stack([v1, v2])    # (2, 3)

        conc = od @ np.linalg.pinv(stain)   # (N, 2)
        max_conc = np.percentile(conc, self.percentile, axis=0)
        return stain, max_conc

    def fit(self, reference_rgb: np.ndarray) -> "MacenkoNormalizer":
        self._stain_matrix_ref, self._max_conc_ref = self._get_stain_matrix(reference_rgb)
        print("Macenko reference stain matrix fitted.")
        return self

    def transform(self, img_rgb: np.ndarray) -> np.ndarray:
        if self._stain_matrix_ref is None:
            raise RuntimeError("Call .fit() before .transform().")
        H, W = img_rgb.shape[:2]
        od = self._rgb_to_od(img_rgb).reshape(-1, 3)
        stain_src, max_conc_src = self._get_stain_matrix(img_rgb)
        conc = od @ np.linalg.pinv(stain_src)
        conc_norm = conc / (max_conc_src + 1e-6) * self._max_conc_ref
        od_norm = conc_norm @ self._stain_matrix_ref
        return self._od_to_rgb(od_norm).reshape(H, W, 3)

class WSIPatcher:

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg

    def _patch_has_tissue(
        self,
        mask_l0: np.ndarray,
        x: int, y: int,
        patch_size_l0: int,
    ) -> bool:
        region = mask_l0[y: y + patch_size_l0, x: x + patch_size_l0]
        return region.size > 0 and (region > 0).mean() >= self.cfg.tissue_threshold

    def extract_and_save(
        self,
        slide: SlideReader,
        mask: np.ndarray,
        thumb_size: tuple[int, int],
        normalizer: Optional[MacenkoNormalizer],
        out_dir: Path,
    ) -> int:
        patch_level = min(self.cfg.patch_level, slide.level_count - 1)
        ds = slide.level_downsamples[patch_level]
        W_pl, H_pl = slide.level_dimensions[patch_level]
        W0, H0     = slide.level_dimensions[0]
        ps     = self.cfg.patch_size
        stride = self.cfg.stride

        patch_size_l0 = int(ps * ds)
        stride_l0     = int(stride * ds)

        # Upscale tissue mask once to level-0 resolution
        print("Upscaling tissue mask to level-0 (%d×%d) …", W0, H0)
        mask_l0 = cv2.resize(mask, (W0, H0), interpolation=cv2.INTER_NEAREST)

        cols = (W_pl - ps) // stride + 1
        rows = (H_pl - ps) // stride + 1
        print(
            "Grid: %d cols × %d rows = %d candidates (level %d, patch %dpx)",
            cols, rows, cols * rows, patch_level, ps,
        )

        patches_dir = out_dir / "patches"
        patches_dir.mkdir(parents=True, exist_ok=True)

        saved = skipped = 0

        with tqdm(total=cols * rows, desc="Patching", unit="patch") as pbar:
            for row in range(rows):
                for col in range(cols):
                    x_pl = col * stride
                    y_pl = row * stride
                    x_l0 = int(x_pl * ds)
                    y_l0 = int(y_pl * ds)
                    pbar.update(1)

                    if not self._patch_has_tissue(mask_l0, x_l0, y_l0, patch_size_l0):
                        skipped += 1
                        continue

                    region = slide.read_region((x_l0, y_l0), patch_level, (ps, ps))
                    patch_rgb = np.array(region.convert("RGB"))

                    if normalizer is not None:
                        try:
                            patch_rgb = normalizer.transform(patch_rgb)
                        except Exception as exc:
                            print("Macenko skipped for patch (%d,%d): %s", col, row, exc)

                    fname = f"patch_r{row:05d}_c{col:05d}.{self.cfg.save_format}"
                    Image.fromarray(patch_rgb).save(str(patches_dir / fname))
                    saved += 1

                    if self.cfg.max_patches and saved >= self.cfg.max_patches:
                        pbar.close()
                        break
                    if saved % 500 == 0:
                        gc.collect()

                if self.cfg.max_patches and saved >= self.cfg.max_patches:
                    break

        print("Patches saved: %d - Background skipped: %d", saved, skipped)
        return saved

class WSIPipeline:

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.out_dir = Path(cfg.output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _auto_reference(
        self, slide: SlideReader, mask: np.ndarray,
        thumb_size: tuple, n_max: int = 30,
    ) -> Optional[np.ndarray]:
        """Pick the first tissue-rich patch as the Macenko reference."""
        patcher = WSIPatcher(self.cfg)
        pl = min(self.cfg.patch_level, slide.level_count - 1)
        ds = slide.level_downsamples[pl]
        W_pl, H_pl = slide.level_dimensions[pl]
        W0, H0 = slide.level_dimensions[0]
        ps = self.cfg.patch_size
        ps_l0 = int(ps * ds)

        mask_l0 = cv2.resize(mask, (W0, H0), interpolation=cv2.INTER_NEAREST)

        found = 0
        for y in range(0, H_pl - ps, ps):
            for x in range(0, W_pl - ps, ps):
                x0, y0 = int(x * ds), int(y * ds)
                if not patcher._patch_has_tissue(mask_l0, x0, y0, ps_l0):
                    continue
                region = slide.read_region((x0, y0), pl, (ps, ps))
                print("Auto-reference patch selected at level 0 (%d, %d)", x0, y0)
                return np.array(region.convert("RGB"))
            found += 1
            if found >= n_max:
                break
        return None

    def run(self) -> None:
        print("WSI Pipeline  —  %s", self.cfg.wsi_path)

        slide = open_slide(self.cfg.wsi_path)
        print(
            "Levels: %d  |  Level-0: %s",
            slide.level_count, slide.level_dimensions[0],
        )

        masker = TissueMasker(self.cfg)
        mask, thumb_size = masker.build_mask(slide)
        if self.cfg.save_mask:
            masker.save_mask(mask, self.out_dir)

        normalizer: Optional[MacenkoNormalizer] = None
        if self.cfg.normalize:
            if self.cfg.reference_patch_path:
                ref_rgb = np.array(Image.open(self.cfg.reference_patch_path).convert("RGB"))
            else:
                ref_rgb = self._auto_reference(slide, mask, thumb_size)

            if ref_rgb is not None:
                normalizer = MacenkoNormalizer(
                    beta=self.cfg.macenko_beta,
                    percentile=self.cfg.macenko_percentile,
                ).fit(ref_rgb)
            else:
                print("Could not find a reference patch")

        print("Patching (level %d)", self.cfg.patch_level)
        patcher = WSIPatcher(self.cfg)
        total = patcher.extract_and_save(slide, mask, thumb_size, normalizer, self.out_dir)

        slide.close()
        print("Done. %d patches -> %s", total, self.out_dir)

def main():
    cfg = PipelineConfig(
        wsi_path="HCM-EXPT-1004-C50-06A-01-S1-HE.8829C2B5-A911-4F3E-A033-F5106997226B.svs",
        output_dir="wsi_patches_output",

        mask_level=2,
        otsu_channel="saturation",

        patch_level=0,
        patch_size=256,
        stride=256,
        tissue_threshold=0.5,

        normalize=True,
        reference_patch_path=None,
        macenko_percentile=99.0,
        macenko_beta=0.15,

        max_patches=None,
        save_mask=True,
        save_format="png",
    )

    pipeline = WSIPipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    main()

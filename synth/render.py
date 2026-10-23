"""Render document HTML to a page image, then degrade copies to look like real uploads.

    from synth.render import Renderer, degrade

Renderer keeps one headless Chromium open; degrade() turns a clean page into a plausible
phone photo or office scan (skew, blur, uneven light, noise, JPEG artefacts).
"""

import io
import random

import numpy as np
from PIL import Image, ImageFilter
from playwright.sync_api import sync_playwright

A4_WIDTH, A4_HEIGHT = 794, 1123  # CSS pixels at 96 dpi
SCALE = 1.5  # render at ~144 dpi so text stays legible after degradation


class Renderer:
    def __enter__(self):
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch()
        self._page = self._browser.new_page(
            viewport={"width": A4_WIDTH, "height": A4_HEIGHT}, device_scale_factor=SCALE
        )
        return self

    def __exit__(self, *exc):
        self._browser.close()
        self._pw.stop()

    def render(self, html: str) -> Image.Image:
        """One A4 page: content beyond the first page is cut off, as on a single scanned sheet."""
        self._page.set_content(html, wait_until="load")
        # Gemini's layouts sometimes run wider than A4; shrink to fit instead of clipping.
        self._page.evaluate(
            f"""() => {{
                const w = document.documentElement.scrollWidth;
                if (w > {A4_WIDTH}) document.documentElement.style.zoom = {A4_WIDTH} / w;
            }}"""
        )
        png = self._page.screenshot(clip={"x": 0, "y": 0, "width": A4_WIDTH, "height": A4_HEIGHT})
        return Image.open(io.BytesIO(png)).convert("RGB")


def degrade(img: Image.Image, rng: random.Random, kind: str | None = None) -> tuple[Image.Image, str]:
    """Return (degraded image, kind) where kind is 'phone' or 'scan'."""
    kind = kind or rng.choice(["phone", "scan"])
    nrng = np.random.default_rng(rng.randrange(2**32))
    w, h = img.size

    if kind == "phone":
        # Paper on a table: tilt, a margin of background, and light falling off one side.
        img = img.rotate(rng.uniform(-4, 4), resample=Image.BICUBIC, expand=True, fillcolor=(120, 110, 100))
        arr = np.asarray(img).astype(np.float32)
        ys, xs = np.mgrid[0 : arr.shape[0], 0 : arr.shape[1]]
        angle = rng.uniform(0, 2 * np.pi)
        ramp = np.cos(angle) * xs / arr.shape[1] + np.sin(angle) * ys / arr.shape[0]
        light = 1 - rng.uniform(0.15, 0.35) * (ramp - ramp.min()) / (np.ptp(ramp) + 1e-6)
        arr = arr * light[..., None]
        arr = arr * rng.uniform(0.9, 1.05) + rng.uniform(-10, 10)  # exposure
        arr += nrng.normal(0, rng.uniform(3, 8), arr.shape)  # sensor noise
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.4, 1.3)))
        img = img.resize((int(img.width * rng.uniform(0.6, 0.85)), int(img.height * rng.uniform(0.6, 0.85))))
        quality = rng.randint(35, 70)
    else:
        # Office scanner: small skew, grey cast, speckle, sometimes bilevel.
        img = img.rotate(rng.uniform(-1.5, 1.5), resample=Image.BICUBIC, expand=False, fillcolor=(255, 255, 255))
        img = img.convert("L")
        arr = np.asarray(img).astype(np.float32)
        arr = arr * rng.uniform(0.85, 1.0) + rng.uniform(0, 20)
        speckle = nrng.random(arr.shape) < rng.uniform(0.0005, 0.003)
        arr[speckle] = rng.uniform(0, 80)
        if rng.random() < 0.3:
            arr = np.where(arr > rng.uniform(150, 190), 255, 0)  # cheap fax/bilevel look
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).convert("RGB")
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 0.7)))
        quality = rng.randint(50, 85)

    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB"), kind

"""
src.evaluation.medium_gate — Embedding/perceptual model gate (~100ms).

Models are LAZY LOADED — imports happen only when a scoring method runs.
If pyiqa or clip are not installed, the gate passes with a warning log.
This module NEVER blocks the pipeline due to missing optional deps.

Supported models (all optional):
  - pyiqa: NIMA, BRISQUE, MUSIQ (aesthetic + quality scoring)
  - clip (openai/clip): text-image alignment, style consistency, novelty

Public API:
  MediumGate.score_aesthetic(image_path)                     -> float
  MediumGate.score_quality(image_path)                       -> float
  MediumGate.score_style_consistency(image_path, brand_emb)  -> float
  MediumGate.score_text_image_alignment(image_path, text)    -> float
  MediumGate.score_novelty(image_path, corpus_emb, k=5)      -> float
  MediumGate.run_all(output, brand_context)                  -> GateResult
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Sentinel: indicates a model was requested but is unavailable.
_UNAVAILABLE = object()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class MediumGateConfig:
    """
    Configuration for the medium gate.

    Parameters
    ----------
    aesthetic_metric : str
        pyiqa metric name for aesthetic scoring ("nima" or "musiq").
    quality_metric : str
        pyiqa metric name for quality scoring ("brisque" or "hyperiqa").
    aesthetic_min : float
        Minimum normalised aesthetic score to pass (0–1).
    quality_min : float
        Minimum normalised quality score to pass (0–1).
    style_min : float
        Minimum cosine similarity to brand embeddings to pass (0–1).
    alignment_min : float
        Minimum CLIPScore text-image alignment to pass (0–1).
    novelty_min : float
        Minimum novelty score — too low = likely duplicate (0–1).
    novelty_max : float
        Maximum novelty score — too high = too weird / off-brand (0–1).
    clip_model : str
        CLIP backbone to use ("ViT-B/32" or "ViT-L/14").
    device : str | None
        Torch device ("cpu", "cuda", "mps").  Auto-detected if None.
    """

    aesthetic_metric: str = "nima"
    quality_metric: str = "brisque"
    aesthetic_min: float = 0.0      # 0 = disabled (pass any)
    quality_min: float = 0.0        # 0 = disabled
    style_min: float = 0.0          # 0 = disabled
    alignment_min: float = 0.0      # 0 = disabled
    novelty_min: float = 0.0        # 0 = disabled
    novelty_max: float = 1.0        # 1 = disabled
    clip_model: str = "ViT-B/32"
    device: str | None = None       # auto-detect


# ---------------------------------------------------------------------------
# MediumGate
# ---------------------------------------------------------------------------


class MediumGate:
    """
    Embedding and perceptual model gate.

    All models are lazy-loaded on first use.  Missing optional libraries
    produce a warning and cause the affected score to be skipped (returns
    None), contributing no failures.

    Parameters
    ----------
    config : MediumGateConfig | None
        Gate configuration.  Defaults if None.
    """

    def __init__(self, config: MediumGateConfig | None = None) -> None:
        self._config = config or MediumGateConfig()
        self._pyiqa: Any = _UNAVAILABLE      # lazy: the pyiqa module
        self._pyiqa_checked = False
        self._clip_model: Any = _UNAVAILABLE  # lazy: CLIP model
        self._clip_preprocess: Any = None
        self._clip_checked = False
        self._torch: Any = _UNAVAILABLE
        self._device: Any = None

    # ------------------------------------------------------------------
    # Public scoring methods
    # ------------------------------------------------------------------

    def score_aesthetic(self, image_path: str) -> float | None:
        """
        Score aesthetic quality using NIMA or MUSIQ via pyiqa.

        Returns a normalised score 0–1, or None if pyiqa is unavailable.
        NIMA raw scores 1–10; normalised as (raw - 1) / 9.
        MUSIQ raw scores 0–100; normalised as raw / 100.
        """
        pyiqa = self._get_pyiqa()
        if pyiqa is _UNAVAILABLE:
            return None

        try:
            torch = self._get_torch()
            device = self._get_device()
            metric = pyiqa.create_metric(self._config.aesthetic_metric, device=device)
            raw = float(metric(image_path))

            m = self._config.aesthetic_metric.lower()
            if "nima" in m:
                return max(0.0, min(1.0, (raw - 1.0) / 9.0))
            else:
                # MUSIQ 0-100
                return max(0.0, min(1.0, raw / 100.0))

        except Exception as exc:
            logger.warning("[MediumGate] score_aesthetic failed: %s", exc)
            return None

    def score_quality(self, image_path: str) -> float | None:
        """
        Score technical quality using BRISQUE (no-reference IQA).

        BRISQUE: lower is better (0 = perfect, 100 = worst).
        Returns normalised 0–1 (higher = better quality).
        Returns None if pyiqa is unavailable.
        """
        pyiqa = self._get_pyiqa()
        if pyiqa is _UNAVAILABLE:
            return None

        try:
            device = self._get_device()
            metric = pyiqa.create_metric("brisque", device=device)
            raw = float(metric(image_path))
            # Clamp to 0-100, invert (lower BRISQUE = higher quality)
            normalised = 1.0 - max(0.0, min(1.0, raw / 100.0))
            return normalised

        except Exception as exc:
            logger.warning("[MediumGate] score_quality failed: %s", exc)
            return None

    def score_style_consistency(
        self,
        image_path: str,
        brand_embeddings: np.ndarray | None,
    ) -> float | None:
        """
        Score style consistency as cosine similarity to brand embeddings.

        Parameters
        ----------
        image_path : str
        brand_embeddings : np.ndarray, shape (N, D) or (D,)
            CLIP embeddings of reference brand images.
            If shape (D,), treated as a single pre-averaged fingerprint.

        Returns
        -------
        float | None
            Cosine similarity 0–1, or None if CLIP unavailable or embeddings empty.
        """
        if brand_embeddings is None or (
            hasattr(brand_embeddings, "__len__") and len(brand_embeddings) == 0
        ):
            logger.debug("[MediumGate] score_style_consistency: no brand embeddings")
            return None

        model, preprocess = self._get_clip()
        if model is _UNAVAILABLE:
            return None

        try:
            torch = self._get_torch()
            device = self._get_device()
            import PIL.Image as PilImage  # PIL is a soft dep

            image = preprocess(PilImage.open(image_path)).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = model.encode_image(image)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            emb = image_features.cpu().numpy().flatten()  # (D,)

            brand_emb = np.array(brand_embeddings)
            if brand_emb.ndim == 2:
                # Average multiple reference embeddings into a fingerprint
                fingerprint = brand_emb.mean(axis=0)
            else:
                fingerprint = brand_emb

            fingerprint = fingerprint / (np.linalg.norm(fingerprint) + 1e-8)
            similarity = float(np.dot(emb, fingerprint))
            return max(0.0, min(1.0, (similarity + 1.0) / 2.0))  # normalise -1..1 → 0..1

        except Exception as exc:
            logger.warning("[MediumGate] score_style_consistency failed: %s", exc)
            return None

    def score_text_image_alignment(
        self,
        image_path: str,
        text_description: str,
    ) -> float | None:
        """
        Score text-image alignment using CLIPScore.

        CLIPScore = max(100 × cos(image_emb, text_emb), 0) / 100, normalised to 0–1.
        Returns None if CLIP is unavailable.
        """
        if not text_description:
            logger.debug("[MediumGate] score_text_image_alignment: no text description")
            return None

        model, preprocess = self._get_clip()
        if model is _UNAVAILABLE:
            return None

        try:
            torch = self._get_torch()
            import clip as clip_lib
            import PIL.Image as PilImage

            device = self._get_device()
            image = preprocess(PilImage.open(image_path)).unsqueeze(0).to(device)
            text = clip_lib.tokenize([text_description[:77]]).to(device)  # CLIP max 77 tokens

            with torch.no_grad():
                image_features = model.encode_image(image)
                text_features = model.encode_text(text)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                similarity = (image_features @ text_features.T).item()

            # CLIPScore: cos sim normalised 0–1
            clip_score = max(0.0, similarity)
            return float(clip_score)

        except Exception as exc:
            logger.warning("[MediumGate] score_text_image_alignment failed: %s", exc)
            return None

    def score_novelty(
        self,
        image_path: str,
        corpus_embeddings: np.ndarray | None,
        k: int = 5,
    ) -> float | None:
        """
        Score novelty as k-NN distance from corpus embeddings.

        Novelty = 1 − mean(top-k cosine similarities to corpus).
        Score 0 = duplicate, 1 = completely novel.

        Returns None if CLIP is unavailable or corpus is empty.
        """
        if corpus_embeddings is None or (
            hasattr(corpus_embeddings, "__len__") and len(corpus_embeddings) == 0
        ):
            logger.debug("[MediumGate] score_novelty: no corpus embeddings — skipping")
            return None

        model, preprocess = self._get_clip()
        if model is _UNAVAILABLE:
            return None

        try:
            torch = self._get_torch()
            import PIL.Image as PilImage

            device = self._get_device()
            image = preprocess(PilImage.open(image_path)).unsqueeze(0).to(device)

            with torch.no_grad():
                image_features = model.encode_image(image)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            emb = image_features.cpu().numpy().flatten()  # (D,)

            corpus = np.array(corpus_embeddings)  # (N, D)
            # Normalise corpus rows
            norms = np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-8
            corpus_norm = corpus / norms

            similarities = corpus_norm @ emb  # (N,)
            top_k = int(min(k, len(similarities)))
            top_sims = np.sort(similarities)[::-1][:top_k]
            novelty = 1.0 - float(np.mean(top_sims))
            return max(0.0, min(1.0, novelty))

        except Exception as exc:
            logger.warning("[MediumGate] score_novelty failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------

    def run_all(
        self,
        output: Any,
        brand_context: dict[str, Any] | None = None,
    ) -> Any:  # returns GateResult
        """
        Run all medium-gate checks and return an aggregate GateResult.

        Parameters
        ----------
        output : CreativeOutput
        brand_context : dict | None
            Optional context with keys:
                brand_embeddings   (np.ndarray): brand CLIP embeddings
                corpus_embeddings  (np.ndarray): past output embeddings
                novelty_k          (int): k for novelty k-NN

        Returns
        -------
        GateResult
        """
        from src.evaluation import GateResult

        t0 = time.perf_counter()
        cfg = self._config
        brand_context = brand_context or {}
        failures: list[str] = []
        scores: list[float] = []

        image_path = output.output_path

        if image_path is None:
            logger.warning(
                "[MediumGate] output.output_path is None — all ML checks skipped"
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return GateResult(passed=True, score=1.0, failures=[], time_ms=elapsed_ms)

        # --- Aesthetic ---
        aesthetic = self.score_aesthetic(image_path)
        if aesthetic is not None:
            scores.append(aesthetic)
            if cfg.aesthetic_min > 0 and aesthetic < cfg.aesthetic_min:
                failures.append(
                    f"Aesthetic score {aesthetic:.3f} below minimum {cfg.aesthetic_min:.3f}"
                )

        # --- Quality ---
        quality = self.score_quality(image_path)
        if quality is not None:
            scores.append(quality)
            if cfg.quality_min > 0 and quality < cfg.quality_min:
                failures.append(
                    f"Quality score {quality:.3f} below minimum {cfg.quality_min:.3f}"
                )

        # --- Style consistency ---
        brand_emb = brand_context.get("brand_embeddings")
        style = self.score_style_consistency(image_path, brand_emb)
        if style is not None:
            scores.append(style)
            if cfg.style_min > 0 and style < cfg.style_min:
                failures.append(
                    f"Style consistency {style:.3f} below minimum {cfg.style_min:.3f}"
                )

        # --- Text-image alignment ---
        text_desc = output.text_description or brand_context.get("text_description")
        if text_desc:
            alignment = self.score_text_image_alignment(image_path, text_desc)
            if alignment is not None:
                scores.append(alignment)
                if cfg.alignment_min > 0 and alignment < cfg.alignment_min:
                    failures.append(
                        f"Text-image alignment {alignment:.3f} below minimum "
                        f"{cfg.alignment_min:.3f}"
                    )

        # --- Novelty ---
        corpus_emb = brand_context.get("corpus_embeddings")
        k = int(brand_context.get("novelty_k", 5))
        novelty = self.score_novelty(image_path, corpus_emb, k=k)
        if novelty is not None:
            if cfg.novelty_min > 0 and novelty < cfg.novelty_min:
                failures.append(
                    f"Novelty score {novelty:.3f} below minimum {cfg.novelty_min:.3f} "
                    f"(possible duplicate)"
                )
            if novelty > cfg.novelty_max:
                failures.append(
                    f"Novelty score {novelty:.3f} above maximum {cfg.novelty_max:.3f} "
                    f"(possibly too far off-brand)"
                )
            # Include novelty in score but it's not a simple higher-is-better metric.
            # Use a clamped sweet-spot value.
            novelty_score = 1.0 - abs(novelty - 0.7) / 0.7  # peak at 0.7
            scores.append(max(0.0, novelty_score))

        passed = len(failures) == 0
        aggregate_score = float(np.mean(scores)) if scores else 1.0  # pass by default

        elapsed_ms = (time.perf_counter() - t0) * 1000

        logger.info(
            "[MediumGate] passed=%s score=%.3f failures=%d time_ms=%.2f",
            passed, aggregate_score, len(failures), elapsed_ms,
        )

        return GateResult(
            passed=passed,
            score=aggregate_score,
            failures=failures,
            time_ms=elapsed_ms,
        )

    # ------------------------------------------------------------------
    # Lazy model loaders
    # ------------------------------------------------------------------

    def _get_torch(self) -> Any:
        """Lazy-load torch. Returns module or raises ImportError."""
        if self._torch is _UNAVAILABLE:
            try:
                import torch as _torch
                self._torch = _torch
                self._device = None  # reset device so it's re-detected with torch
            except ImportError:
                logger.warning(
                    "[MediumGate] torch not installed — ML scoring unavailable. "
                    "Install with: pip install torch"
                )
                raise
        return self._torch

    def _get_device(self) -> Any:
        """Return torch device, auto-detecting if needed."""
        if self._device is None:
            torch = self._get_torch()
            if self._config.device:
                self._device = torch.device(self._config.device)
            elif torch.cuda.is_available():
                self._device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._device = torch.device("mps")
            else:
                self._device = torch.device("cpu")
            logger.info("[MediumGate] Using device: %s", self._device)
        return self._device

    def _get_pyiqa(self) -> Any:
        """Lazy-load pyiqa. Returns module or _UNAVAILABLE sentinel."""
        if not self._pyiqa_checked:
            self._pyiqa_checked = True
            try:
                import pyiqa as _pyiqa
                self._pyiqa = _pyiqa
                logger.info("[MediumGate] pyiqa loaded successfully")
            except ImportError:
                self._pyiqa = _UNAVAILABLE
                logger.warning(
                    "[MediumGate] pyiqa not installed — aesthetic/quality scoring skipped. "
                    "Install with: pip install pyiqa (optional)"
                )
        return self._pyiqa

    def _get_clip(self) -> tuple[Any, Any]:
        """
        Lazy-load CLIP model and preprocessor.
        Returns (model, preprocess) or (_UNAVAILABLE, None).
        """
        if not self._clip_checked:
            self._clip_checked = True
            try:
                torch = self._get_torch()
                import clip as _clip
                device = self._get_device()
                model, preprocess = _clip.load(self._config.clip_model, device=device)
                self._clip_model = model
                self._clip_preprocess = preprocess
                logger.info(
                    "[MediumGate] CLIP %s loaded on %s",
                    self._config.clip_model, device,
                )
            except ImportError:
                self._clip_model = _UNAVAILABLE
                self._clip_preprocess = None
                logger.warning(
                    "[MediumGate] clip not installed — alignment/novelty/style scoring skipped. "
                    "Install with: pip install clip-by-openai (optional)"
                )
            except Exception as exc:
                self._clip_model = _UNAVAILABLE
                self._clip_preprocess = None
                logger.warning("[MediumGate] CLIP load failed: %s", exc)
        return self._clip_model, self._clip_preprocess

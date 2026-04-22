# Creative Evaluation Methods

_Research compiled: 2026-04-22 | Atwater / Cheezfish project_

---

## Overview

This document covers automated scoring methods for creative content (ads, visuals, branded materials). Priority is given to **locally runnable** tools — Cheezfish uses local models and cloud-only solutions are lower priority.

---

## 1. Available Automated Scoring Tools / Models

### 1.1 IQA-PyTorch (`pyiqa`) — THE local IQA toolkit
**Repo:** https://github.com/chaofengc/IQA-PyTorch  
**Install:** `pip install pyiqa`

The most comprehensive local IQA toolkit. Pure PyTorch. GPU-accelerated. Includes 50+ metrics.

**Key no-reference (NR) metrics — run without a reference image:**

| Metric | Type | What it measures | Notes |
|--------|------|-----------------|-------|
| `brisque` | NR | Perceptual quality (naturalness) | CPU-fast, classic |
| `niqe` | NR | Naturalness Image Quality | Unsupervised |
| `nrqm` (Ma) | NR | Perceptual quality | Strong on photos |
| `musiq` | NR | Multi-scale quality (Google) | Better on diverse images |
| `nima` | Aesthetic/NR | Neural Image Assessment (AVA) | Trained on human aesthetic ratings |
| `nima-vgg16-ava` | Aesthetic/NR | VGG16 version of NIMA | Lighter |
| `hyperiqa` | NR | Distortion-agnostic quality | Good general purpose |
| `dbcnn` | NR | Deep Bilinear CNN quality | Strong benchmark results |
| `topiq_nr` | NR | Top-down semantic + distortion | 2024, state-of-art |
| `qualiclip` | NR | CLIP-based quality (2025) | Zero-shot capable |

**Key full-reference (FR) metrics — compare against a reference:**

| Metric | What it measures |
|--------|-----------------|
| `lpips` | Perceptual similarity (VGG/AlexNet) |
| `ssim` / `ssimc` | Structural similarity |
| `psnr` | Signal-level quality |

**Usage example:**
```python
import pyiqa, torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Aesthetic quality (no reference needed)
nima = pyiqa.create_metric('nima', device=device)
score = nima('path/to/image.png')  # returns float 0-10

# Perceptual quality
musiq = pyiqa.create_metric('musiq', device=device)
score = musiq('path/to/image.png')

# Style similarity vs reference
lpips = pyiqa.create_metric('lpips', device=device)
dist = lpips('generated.png', 'reference.png')  # lower = more similar
```

**Locally runnable:** Yes — all models download from HuggingFace on first use, run fully local after. CPU works, GPU recommended for batches.

---

### 1.2 LAION Aesthetic Predictor
**Repo:** https://github.com/LAION-AI/aesthetic-predictor

A lightweight linear model on top of CLIP embeddings. Trained on human aesthetic ratings from LAION datasets.

```python
import torch
import clip

# Load CLIP + linear aesthetic head
model, preprocess = clip.load("ViT-L/14", device=device)
aesthetic_model = get_aesthetic_model(clip_model="vit_l_14")

# Extract CLIP embedding → aesthetic score 1-10
image_features = model.encode_image(preprocessed_image)
score = aesthetic_model(image_features)
```

**Pros:** Very fast (CLIP is already loaded for other metrics). Scores correlate well with human aesthetic preference.  
**Cons:** Trained on general web images, may not align perfectly with specific brand aesthetics. Use as one signal among many.

**Locally runnable:** Yes — model is ~50MB, runs on CPU in <100ms per image.

---

### 1.3 CLIP / CLIPScore
**Paper:** Radford et al., "Learning Transferable Visual Models From Natural Language Supervision" (arXiv:2103.00020, OpenAI)  
**CLIPScore paper:** Hessel et al., "CLIPScore: A Reference-free Evaluation Metric for Image Captioning" (arXiv:2104.08718, EMNLP 2021)

CLIP encodes images and text into a shared embedding space. CLIPScore measures how well an image matches a text description — without needing a reference image.

```python
import torch
import clip

model, preprocess = clip.load("ViT-B/32", device=device)

# Image-text alignment score
image = preprocess(Image.open("ad.png")).unsqueeze(0)
text = clip.tokenize(["a clean modern advertisement for coffee brand"])

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    similarity = (image_features @ text_features.T).item()  # -1 to 1
```

**CLIPScore formula:**
```
CLIPScore(image, caption) = max(100 × cos(image_emb, text_emb), 0)
```

**Applications for Atwater:**
- **Brief alignment:** Does the output match the creative brief?
- **Brand alignment:** Does the image match brand descriptor text?
- **Style consistency:** Are two images in the same style? (image-to-image cosine sim)

**Locally runnable:** Yes — CLIP ViT-B/32 is ~340MB, ViT-L/14 is ~890MB. Runs on CPU, fast on GPU.

---

### 1.4 Q-Align (LMM-based Quality Scoring)
**Paper:** Wu et al., "Q-Align: Teaching LMMs for Visual Scoring via Discrete Text-Defined Levels" (ICML 2024)

Uses a Large Multimodal Model fine-tuned to give discrete quality ratings (bad/poor/fair/good/excellent). Also available in pyiqa.

**Locally runnable:** Requires a 7B+ LMM. Works locally on M2/M3 Max or better with sufficient RAM (≥32GB). Slower but more semantically rich than CNN-based metrics.

---

## 2. What Can Run Locally (Cheezfish Context)

### Tier 1: Fast & Always-On (< 100ms per image, CPU)
| Tool | What | RAM needed |
|------|------|-----------|
| BRISQUE | Perceptual quality | <500MB |
| NIQE | Naturalness | <500MB |
| SSIM/PSNR | Structural comparison | <100MB |
| CLIP ViT-B/32 + CLIPScore | Text-image alignment | ~2GB VRAM or 4GB RAM |
| LAION Aesthetic Predictor | Aesthetic score | ~2GB (CLIP already loaded) |
| Colour analysis (PIL/OpenCV) | Palette, contrast | No ML needed |
| Typography rule-checking | Font size, line height | No ML needed |

### Tier 2: Quality-Optimised (100ms–2s, GPU recommended)
| Tool | What | RAM needed |
|------|------|-----------|
| NIMA (pyiqa) | Human-rated aesthetic quality | ~4GB |
| MUSIQ (pyiqa) | Multi-scale quality | ~4GB |
| LPIPS | Perceptual similarity | ~2GB |
| CLIP ViT-L/14 | Higher-quality CLIP | ~4GB VRAM |
| TOPIQ-NR | State-of-art NR quality | ~6GB |

### Tier 3: Heavy (>2s, strong GPU required)
| Tool | What | Requirement |
|------|------|------------|
| Q-Align | LMM-based semantic quality | 7B model, 32GB+ RAM |
| DINO v2 features | Rich visual features for comparison | ~4GB VRAM |

---

## 3. Rubric Design Patterns

### 3.1 Hierarchical Weighted Rubric

Design rubrics with 3 levels of hierarchy:

```
Level 1: Campaign Score (0-100)
├── Level 2: Dimension Scores (weighted)
│   ├── Visual Quality (weight: 0.25)
│   │   ├── Aesthetic appeal (NIMA score)
│   │   ├── Composition balance (rule-based)
│   │   └── Technical quality (BRISQUE)
│   ├── Brand Alignment (weight: 0.30)
│   │   ├── Colour compliance (palette distance)
│   │   ├── Logo presence & placement (detection)
│   │   └── Brand voice match (CLIPScore vs brand descriptor)
│   ├── Creative Strength (weight: 0.25)
│   │   ├── Concept novelty (embedding distance from corpus)
│   │   ├── Message clarity (LLM judge)
│   │   └── Emotional resonance (CLIP vs emotion text probes)
│   └── Technical Execution (weight: 0.20)
│       ├── Typography quality (size hierarchy, contrast)
│       ├── Colour contrast (WCAG ratio)
│       └── Format compliance (size, resolution, format)
```

### 3.2 Rubric Calibration with Anchor Examples
For each dimension, define 3 anchor examples:
- **Floor example** (score 20-30): intentionally bad output
- **Baseline example** (score 60-70): acceptable but unremarkable
- **Ceiling example** (score 90+): best-in-class reference

Include anchors in every evaluator agent prompt for consistency.

### 3.3 Adaptive Rubric Weights by Campaign Type

```python
RUBRIC_PROFILES = {
    "brand_awareness": {"visual": 0.30, "brand": 0.40, "creative": 0.20, "technical": 0.10},
    "direct_response": {"visual": 0.20, "brand": 0.25, "creative": 0.30, "technical": 0.25},
    "social_organic": {"visual": 0.35, "brand": 0.20, "creative": 0.35, "technical": 0.10},
    "display_ad": {"visual": 0.25, "brand": 0.30, "creative": 0.20, "technical": 0.25},
}
```

Campaign type is parsed from the brief; rubric weights are selected automatically.

### 3.4 Scoring Normalisation
Raw metric outputs are on different scales (BRISQUE 0-100 lower-better; NIMA 1-10 higher-better). Normalise everything to 0-1:

```python
def normalise_score(raw, metric_name):
    """Normalise any metric to 0-1 higher-better."""
    cfg = METRIC_CONFIG[metric_name]  # min, max, lower_better
    normalised = (raw - cfg.min) / (cfg.max - cfg.min)
    return 1 - normalised if cfg.lower_better else normalised
```

---

## 4. Novelty Detection Approaches

### 4.1 Embedding Distance from Corpus
The simplest reliable novelty metric:

```python
# Maintain a corpus of past outputs as CLIP embeddings
past_embeddings = load_corpus_embeddings()  # shape: (N, 512)

# Embed new output
new_embedding = clip_embed(new_image)  # shape: (512,)

# Novelty = distance to nearest neighbour in corpus
similarities = cosine_similarity(new_embedding, past_embeddings)
novelty_score = 1 - similarities.max()  # 0 = duplicate, 1 = completely novel
```

**Threshold:** Similarity > 0.92 = likely duplicate/plagiarism. Similarity < 0.60 = too weird. Sweet spot: 0.65–0.85.

### 4.2 k-NN Novelty (Top-k Distance)
More robust: average distance to top-k nearest neighbours rather than just nearest-1. Handles the case where a single outlier is nearby but the cluster is far.

```python
top_k_sims = sorted(similarities, reverse=True)[:5]
novelty_score = 1 - np.mean(top_k_sims)
```

### 4.3 Semantic Novelty via Text Embeddings
For concept-level novelty (brief-to-concept diversity):

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')  # local, ~80MB

concept_embeddings = model.encode(past_concepts)
new_concept_emb = model.encode(new_concept)
novelty = 1 - cosine_similarity([new_concept_emb], concept_embeddings).max()
```

**Locally runnable:** Yes — SentenceTransformers models are small and fast on CPU.

### 4.4 FID Alternatives for Small Batches

FID requires large sample sizes (thousands) to be reliable. For small batches (10-100 samples):

| Metric | Min samples | Notes |
|--------|------------|-------|
| FID (standard) | 2000+ | Unreliable for small N |
| KID (Kernel Inception Distance) | 50+ | Better for small N, uses U-statistic |
| SFID (Spatial FID) | 50+ | Preserves spatial info, added to pyiqa Jun 2025 |
| DINO cosine similarity | 1+ | No distribution needed; compare individual pairs |
| LPIPS | 1+ | Single-pair perceptual distance |

**Recommendation for Cheezfish:** Use LPIPS and CLIP cosine similarity for single-pair comparisons. Use KID (available in pyiqa) for batch-level distribution comparison with 50+ samples. Avoid FID entirely in production.

```python
# KID via pyiqa (better for small batches)
kid = pyiqa.create_metric('kid')
score = kid('./generated_dir/', './reference_dir/')

# SFID (added Jun 2025 to pyiqa)
sfid = pyiqa.create_metric('sfid')
score = sfid('./generated_dir/', './reference_dir/')
```

---

## 5. Style Consistency Measurement

### 5.1 CLIP-Based Style Fingerprinting

Define a brand's "style fingerprint" as the mean CLIP embedding of 20-50 on-brand reference images:

```python
reference_images = load_brand_references()  # 20-50 approved brand images
ref_embeddings = [clip_embed(img) for img in reference_images]
brand_fingerprint = np.mean(ref_embeddings, axis=0)
brand_fingerprint /= np.linalg.norm(brand_fingerprint)  # normalise

# Score new output for brand style consistency
def style_consistency_score(new_image):
    new_emb = clip_embed(new_image)
    return cosine_similarity(new_emb, brand_fingerprint)
```

**Threshold calibration:** Run on known on-brand and off-brand examples to set your accept/reject threshold empirically.

### 5.2 Gram Matrix Style Similarity (CNN Features)
Classic neural style transfer approach — compares feature correlation patterns between images:

```python
# Using VGG16 features from pyiqa or torchvision
# Gram matrix captures texture/style rather than content
gram_a = gram_matrix(vgg_features(image_a))
gram_b = gram_matrix(vgg_features(image_b))
style_dist = F.mse_loss(gram_a, gram_b)
```

More sensitive to texture/colour/pattern than CLIP. Use when style consistency is about visual texture rather than semantic concept.

### 5.3 Colour Palette Consistency

```python
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans

def extract_palette(image_path, n_colors=5):
    img = Image.open(image_path).convert('RGB')
    pixels = np.array(img).reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_colors, random_state=0)
    kmeans.fit(pixels)
    return kmeans.cluster_centers_  # RGB centroids

def palette_distance(palette_a, palette_b):
    """Earth Mover's Distance approximation for colour palettes."""
    # Simple: min assignment distance between palettes
    from scipy.optimize import linear_sum_assignment
    cost = np.linalg.norm(palette_a[:, None] - palette_b[None, :], axis=-1)
    row_ind, col_ind = linear_sum_assignment(cost)
    return cost[row_ind, col_ind].mean()
```

**Brand compliance check:** Compare extracted palette against brand colour guidelines. Flag outputs where no dominant colour is within Δ20 (in LAB colour space) of any brand colour.

### 5.4 Typography Evaluation (Automated)

Rule-based checks (no ML required):
```python
def check_typography(image, config):
    results = {}
    
    # 1. Contrast ratio (WCAG 2.1)
    text_color, bg_color = detect_text_regions(image)
    results['contrast_ratio'] = wcag_contrast_ratio(text_color, bg_color)
    results['contrast_pass'] = results['contrast_ratio'] >= 4.5  # AA standard
    
    # 2. Font size hierarchy (heading > body > caption)
    # Detected via OCR + bounding box analysis
    text_regions = ocr_with_boxes(image)
    results['size_hierarchy'] = check_hierarchy(text_regions)
    
    # 3. Text legibility (BRISQUE on text region crop)
    text_region = crop_text_region(image, text_regions)
    results['legibility'] = brisque_score(text_region)
    
    return results
```

Tools: `pytesseract` (OCR), `PIL` (colour), `pyiqa` (perceptual quality), `pyvips` (image processing).

---

## 6. Brand Alignment Measurement

### 6.1 Multi-Probe CLIPScore
Rather than one text probe, use multiple brand descriptor prompts and average:

```python
brand_probes = [
    "a professional advertisement for [Brand]",
    "clean, modern, premium design with [Brand Colours]",
    "[Brand] brand visual identity",
    "sophisticated marketing material consistent with [Brand] style",
]

scores = []
for probe in brand_probes:
    score = clip_score(image, probe)
    scores.append(score)

brand_alignment = np.mean(scores)
```

More robust than a single probe. Captures different aspects of brand identity.

### 6.2 Visual Element Detection for Brand Compliance
For logos and brand elements, use a fine-tuned object detection model:
- **YOLOv8** (ultralytics, local): train on brand asset examples
- **DINO / GroundingDINO** (local): zero-shot detection with text prompts

```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')  # or fine-tuned brand model

results = model('ad.png')
# Check if brand logo is present and in correct position
logo_detected = any(cls == 'brand_logo' for cls in results.boxes.cls)
logo_placement = results.boxes.xyxy if logo_detected else None
```

### 6.3 Colour Compliance Scoring
```python
BRAND_COLORS_LAB = [
    (50, 12, -45),   # primary blue (LAB)
    (95, -3, 7),     # primary white
    (20, 0, 0),      # primary black
]

def colour_compliance(image, brand_colors, tolerance=20):
    palette = extract_palette(image, n_colors=8)
    palette_lab = rgb_to_lab(palette)
    
    # Each dominant colour should be within tolerance of a brand colour
    compliance_score = 0
    for color in palette_lab:
        min_dist = min(delta_e(color, bc) for bc in brand_colors)
        if min_dist <= tolerance:
            compliance_score += 1
    
    return compliance_score / len(palette_lab)
```

---

## 7. Layout Quality Scoring

### 7.1 Rule-Based Layout Analysis

```python
def score_layout(image):
    h, w = image.shape[:2]
    
    scores = {}
    
    # Rule of thirds grid compliance
    # (key elements should align with third-lines)
    thirds_h, thirds_w = h/3, w/3
    key_regions = detect_salient_regions(image)
    scores['rule_of_thirds'] = check_thirds_alignment(key_regions, thirds_h, thirds_w)
    
    # Visual balance (left/right, top/bottom weight distribution)
    left_weight = np.mean(image[:, :w//2])
    right_weight = np.mean(image[:, w//2:])
    scores['balance'] = 1 - abs(left_weight - right_weight) / 255
    
    # White space ratio (clutter score)
    non_empty = np.count_nonzero(np.std(image, axis=2) > 10)
    scores['breathing_room'] = 1 - (non_empty / (h * w))
    
    # Focal point clarity (is there a clear dominant region?)
    saliency = compute_saliency(image)
    scores['focal_clarity'] = saliency.max() / saliency.mean()
    
    return scores
```

### 7.2 Saliency Map Alignment
Use `pysaliency` or OpenCV's saliency module to detect where attention goes:
- Does attention land on brand name / CTA?
- Is the visual hierarchy correct (hero → message → CTA)?

---

## 8. Colour Theory Scoring

### 8.1 Harmony Scoring (HSV-Based)

```python
def colour_harmony_score(image):
    """Score colour harmony using HSV analysis."""
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hues = img_hsv[:,:,0].flatten()  # 0-180 in OpenCV
    
    # Extract dominant hues (histogram peak detection)
    hist = np.histogram(hues, bins=36)[0]
    peaks = find_peaks(hist)
    
    # Check against known harmony patterns
    peak_hues = [p * 5 for p in peaks]  # convert bin to degrees
    
    harmony_types = {
        'complementary': is_complementary(peak_hues),   # ~180° apart
        'analogous': is_analogous(peak_hues),            # <30° apart
        'triadic': is_triadic(peak_hues),                # ~120° apart
        'monochromatic': is_monochromatic(peak_hues),    # same hue, different sat/val
    }
    
    # Any known harmony pattern = high score
    return 1.0 if any(harmony_types.values()) else 0.5
```

### 8.2 Contrast Adequacy (LAB Colour Space)
```python
# Delta-E contrast between text and background
from colormath.color_diff import delta_e_cie2000
contrast = delta_e_cie2000(text_lab, bg_lab)
# <10: poor contrast | 10-30: moderate | >30: good contrast
```

---

## 9. A/B Testing Framework for Creative Content

### 9.1 Automated Score-Based Selection
Without user exposure, use automated metrics as proxy signal:

```python
def ab_score_comparison(variant_a, variant_b, rubric):
    score_a = evaluate(variant_a, rubric)
    score_b = evaluate(variant_b, rubric)
    
    # Statistical test (Welch's t-test if N > 30, else permutation test)
    if has_sufficient_samples(score_a, score_b):
        stat, p_value = welchs_t_test(score_a.history, score_b.history)
        winner = 'A' if score_a.mean > score_b.mean and p_value < 0.05 else 'B'
    else:
        winner = 'A' if score_a.mean > score_b.mean else 'B'
    
    return winner, score_a, score_b
```

### 9.2 Multi-Armed Bandit for Variant Selection
Instead of hard A/B (50/50 split), use Thompson Sampling to route more generations to the currently winning strategy:

```python
# Beta distribution for each strategy
alphas = {strategy: 1 for strategy in strategies}  # successes
betas = {strategy: 1 for strategy in strategies}   # failures

def select_strategy():
    samples = {s: np.random.beta(alphas[s], betas[s]) for s in strategies}
    return max(samples, key=samples.get)

def update_strategy(strategy, score, threshold=0.75):
    if score >= threshold:
        alphas[strategy] += 1
    else:
        betas[strategy] += 1
```

---

## 10. Recommended Local Stack for Atwater/Cheezfish

```
Layer 1: Fast quality gates (CPU, <100ms each)
├── BRISQUE (pyiqa) — technical quality floor
├── CLIP ViT-B/32 — brief alignment, style consistency, brand alignment
├── LAION Aesthetic Predictor — human-aesthetic proxy
├── Colour analysis (PIL + sklearn) — palette compliance
└── Typography rules (pytesseract + PIL) — contrast, hierarchy

Layer 2: Quality optimisation signal (GPU, 100ms-2s each)
├── NIMA (pyiqa) — human-rated aesthetic quality (Optuna objective)
├── MUSIQ (pyiqa) — robust multi-scale quality
├── LPIPS (pyiqa) — style/perceptual distance to references
└── CLIP ViT-L/14 — higher quality embeddings for scoring

Layer 3: Deep semantic scoring (GPU, 2-10s, optional)
├── LLM judge (local Ollama: Llama3/Mistral) — holistic coherence
├── TOPIQ-NR (pyiqa) — state-of-art semantic quality
└── Q-Align (if 32GB+ RAM available) — LMM quality rating
```

**Install all of Layer 1 + 2:**
```bash
pip install pyiqa torch torchvision clip-by-openai sentence-transformers scikit-learn pytesseract Pillow colormath scipy
```

---

## References

| Tool / Paper | Source | Year |
|-------------|--------|------|
| IQA-PyTorch (pyiqa) | github.com/chaofengc/IQA-PyTorch | 2022-2025 |
| LAION Aesthetic Predictor | github.com/LAION-AI/aesthetic-predictor | 2022 |
| CLIP | arXiv:2103.00020 (Radford et al.) | 2021 |
| CLIPScore | arXiv:2104.08718 (Hessel et al.) | 2021 |
| NIMA | Google Research | 2018 |
| MUSIQ | Google Research | 2021 |
| TOPIQ | arXiv (Chen et al.) | 2024 |
| Q-Align | ICML 2024 (Wu et al.) | 2024 |
| KID (Kernel Inception Distance) | Demystifying MMD GANs | 2018 |
| SFID (Spatial FID) | pyiqa changelog | 2025 |

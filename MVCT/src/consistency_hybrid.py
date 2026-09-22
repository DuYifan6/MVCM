import hashlib
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


logger = logging.getLogger("ConsistencyBuilderHybrid")


class ConsistencyBuilderHybrid:
    """Efficient four-channel MVCM with batched embeddings and optional NLI."""

    # Bump the signature whenever feature construction changes.  Version 2.1
    # also prevents caches produced by an incomplete SentenceTransformer
    # snapshot (which silently fell back to generic mean pooling) from reuse.
    VERSION = "hybrid-mvcm-v2.1"
    MAX_TRIPLES_PER_SENT = 3
    MAX_TRIPLES_PER_VIEW = 30
    MAX_SENTS_PER_VIEW = 8

    def __init__(
        self,
        sbert_model="models/all-mpnet-base-v2",
        stanford_url="http://localhost:9000",
        nli_model="models/nli-model",
        use_nli=False,
        nli_batch_size=16,
        nli_max_length=256,
        nli_contradiction_id=None,
        sbert_batch_size=64,
        openie_workers=3,
        gpu_batch_auto_reduce=True,
        sem_weight=0.4,
        fact_weight=0.25,
        logic_weight=0.2,
        local_weight=0.15,
        fact_component_weights=(0.4, 0.3, 0.3),
        logic_weights=(0.55, 0.15, 0.15, 0.15),
        alignment_threshold=0.55,
        coverage_target=5,
        max_logic_pairs=12,
        device=None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sbert_model_name = str(sbert_model)
        self.sbert_batch_size = int(sbert_batch_size)
        self.gpu_batch_auto_reduce = bool(gpu_batch_auto_reduce)
        try:
            from pycorenlp import StanfordCoreNLP
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise ImportError(
                "MVCM construction requires sentence-transformers and pycorenlp. "
                "Install requirements.txt in the cloud environment."
            ) from error
        sbert_path = Path(self.sbert_model_name)
        if not sbert_path.is_dir():
            raise FileNotFoundError(
                f"SentenceTransformer model directory was not found: {sbert_path}"
            )
        modules_path = sbert_path / "modules.json"
        if not modules_path.is_file():
            raise FileNotFoundError(
                "Incomplete SentenceTransformer snapshot: "
                f"{modules_path} is missing. Download the complete "
                "sentence-transformers/all-mpnet-base-v2 repository."
            )
        # Sentence-Transformers 2.7.0 does not accept local_files_only here.
        # Set HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 before launching
        # Python when running with downloaded model snapshots.
        self.sbert = SentenceTransformer(str(sbert_path), device=self.device)
        self.nlp = StanfordCoreNLP(stanford_url)
        self.openie_workers = max(1, int(openie_workers))
        self._openie_pool = (
            ThreadPoolExecutor(max_workers=self.openie_workers)
            if self.openie_workers > 1
            else None
        )

        self.use_nli = bool(use_nli)
        self.nli_model_name = str(nli_model)
        self.nli_batch_size = int(nli_batch_size)
        self.nli_max_length = int(nli_max_length)
        self.nli_tokenizer = None
        self.nli_model = None
        self.nli_contradiction_id = nli_contradiction_id
        if self.use_nli:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            if not Path(self.nli_model_name).exists():
                raise FileNotFoundError(
                    f"NLI model was enabled but not found: {self.nli_model_name}"
                )
            self.nli_tokenizer = AutoTokenizer.from_pretrained(
                self.nli_model_name, local_files_only=True
            )
            self.nli_model = AutoModelForSequenceClassification.from_pretrained(
                self.nli_model_name, local_files_only=True
            ).to(self.device)
            self.nli_model.eval()
            self.nli_contradiction_id = self._resolve_contradiction_id(
                self.nli_contradiction_id
            )

        self.mvc_weights = np.asarray(
            [sem_weight, fact_weight, logic_weight, local_weight], dtype=np.float32
        )
        if not np.isclose(self.mvc_weights.sum(), 1.0):
            raise ValueError("MVC channel weights must sum to one")
        self.fact_component_weights = np.asarray(fact_component_weights, dtype=np.float32)
        self.fact_component_weights /= self.fact_component_weights.sum()
        self.logic_weights = np.asarray(logic_weights, dtype=np.float32)
        if len(self.logic_weights) != 4:
            raise ValueError("logic_weights must be (NLI, negation, temporal, numeric)")
        self.alignment_threshold = float(alignment_threshold)
        self.coverage_target = max(1, int(coverage_target))
        self.max_logic_pairs = max(1, int(max_logic_pairs))
        self.cache_signature = self._make_cache_signature(stanford_url)

    def _make_cache_signature(self, stanford_url):
        payload = {
            "version": self.VERSION,
            "sbert": self.sbert_model_name,
            "stanford": str(stanford_url),
            "nli": self.nli_model_name if self.use_nli else None,
            "mvc_weights": self.mvc_weights.tolist(),
            "fact_weights": self.fact_component_weights.tolist(),
            "logic_weights": self.logic_weights.tolist(),
            "alignment_threshold": self.alignment_threshold,
            "coverage_target": self.coverage_target,
            "max_logic_pairs": self.max_logic_pairs,
            "openie_workers": self.openie_workers,
            "limits": [
                self.MAX_TRIPLES_PER_SENT,
                self.MAX_TRIPLES_PER_VIEW,
                self.MAX_SENTS_PER_VIEW,
            ],
        }
        serialized = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        return f"{self.VERSION}-{hashlib.sha256(serialized.encode()).hexdigest()[:16]}"

    def _resolve_contradiction_id(self, configured_id):
        if configured_id is not None:
            return int(configured_id)
        labels = getattr(self.nli_model.config, "id2label", {})
        for index, label in labels.items():
            if "contrad" in str(label).lower():
                return int(index)
        raise ValueError(
            "Cannot infer the NLI contradiction label. Set NLI_CONTRADICTION_ID explicitly."
        )

    @staticmethod
    def _normalize_text(text):
        return re.sub(r"\W+", " ", str(text or "").lower()).strip()

    @staticmethod
    def _token_jaccard(left, right):
        left_tokens = set(ConsistencyBuilderHybrid._normalize_text(left).split())
        right_tokens = set(ConsistencyBuilderHybrid._normalize_text(right).split())
        if not left_tokens or not right_tokens:
            return 0.0
        return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)

    def _encode(self, texts: Sequence[str]):
        if not texts:
            return torch.empty((0, 0), device=self.device)
        batch_size = min(self.sbert_batch_size, len(texts))
        while True:
            try:
                with torch.inference_mode():
                    return self.sbert.encode(
                        list(texts),
                        batch_size=batch_size,
                        convert_to_tensor=True,
                        normalize_embeddings=True,
                        show_progress_bar=False,
                    )
            except RuntimeError as error:
                is_oom = "out of memory" in str(error).lower()
                if not (self.gpu_batch_auto_reduce and is_oom and batch_size > 1):
                    raise
                batch_size = max(1, batch_size // 2)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.warning("SBERT OOM; reducing batch size to %d", batch_size)

    def extract_openie_triples(self, text: str) -> List[Dict[str, str]]:
        try:
            annotation = self.nlp.annotate(
                str(text),
                properties={
                    "annotators": "openie,ner,tokenize,ssplit",
                    "outputFormat": "json",
                },
            )
            if isinstance(annotation, str):
                annotation = json.loads(annotation)
        except Exception as error:
            logger.warning("OpenIE failed: %s", error)
            return []

        triples = []
        for sentence in annotation.get("sentences", []):
            sentence_text = " ".join(
                token.get("word", "") for token in sentence.get("tokens", [])
            ).strip()
            for extraction in sentence.get("openie", [])[: self.MAX_TRIPLES_PER_SENT]:
                triples.append(
                    {
                        "sentence": sentence_text,
                        "subject": extraction.get("subject", ""),
                        "relation": extraction.get("relation", ""),
                        "object": extraction.get("object", ""),
                    }
                )
                if len(triples) >= self.MAX_TRIPLES_PER_VIEW:
                    return triples
        return triples

    def semantic_matrix(self, views):
        embeddings = self._encode(views)
        similarities = embeddings @ embeddings.T
        return ((similarities + 1.0) / 2.0).clamp(0.0, 1.0).cpu().numpy()

    def _triple_component_embeddings(self, triples_per_view):
        texts = []
        indices = {}
        for view_index, triples in enumerate(triples_per_view):
            for triple_index, triple in enumerate(triples):
                for component in ("subject", "relation", "object"):
                    indices[(view_index, triple_index, component)] = len(texts)
                    texts.append(self._normalize_text(triple.get(component, "")) or "[EMPTY]")
        embeddings = self._encode(texts)
        return embeddings, indices

    def _affinity_matrix(self, view_i, view_j, triples_per_view, embeddings, indices):
        triples_i = triples_per_view[view_i]
        triples_j = triples_per_view[view_j]
        if not triples_i or not triples_j:
            return np.zeros((len(triples_i), len(triples_j)), dtype=np.float32)

        component_matrices = []
        for component in ("subject", "relation", "object"):
            index_i = [indices[(view_i, row, component)] for row in range(len(triples_i))]
            index_j = [indices[(view_j, row, component)] for row in range(len(triples_j))]
            cosine = embeddings[index_i] @ embeddings[index_j].T
            component_matrices.append(((cosine + 1.0) / 2.0).clamp(0.0, 1.0))
        affinity = sum(
            float(weight) * matrix
            for weight, matrix in zip(self.fact_component_weights, component_matrices)
        )
        return affinity.cpu().numpy().astype(np.float32)

    @staticmethod
    def _bidirectional_best_alignment(affinity):
        if affinity.size == 0:
            return 0.0
        return float(0.5 * (affinity.max(axis=1).mean() + affinity.max(axis=0).mean()))

    def _candidate_pairs(self, affinity):
        if affinity.size == 0:
            return []
        candidates = [
            (float(affinity[row, column]), row, column)
            for row, column in zip(*np.where(affinity >= self.alignment_threshold))
        ]
        candidates.sort(reverse=True)
        return candidates[: self.max_logic_pairs]

    @staticmethod
    def _negation_conflict(text_a, text_b):
        pattern = re.compile(
            r"\b(?:no|not|never|neither|nor|without|deny|denied|denies|false|failed|refused)\b",
            re.IGNORECASE,
        )
        return float(bool(pattern.search(text_a)) != bool(pattern.search(text_b)))

    @staticmethod
    def _temporal_conflict(text_a, text_b) -> Optional[float]:
        pattern = re.compile(r"\b(?:19|20)\d{2}\b")
        years_a = set(pattern.findall(text_a))
        years_b = set(pattern.findall(text_b))
        if not years_a or not years_b:
            return None
        return float(years_a.isdisjoint(years_b))

    @staticmethod
    def _quantities(text):
        pattern = re.compile(
            r"(?<!\w)([-+]?\d+(?:,\d{3})*(?:\.\d+)?)\s*"
            r"(billion|million|thousand|percent|%|bn|m|k)?\b",
            re.IGNORECASE,
        )
        scales = {
            "billion": 1e9,
            "bn": 1e9,
            "million": 1e6,
            "m": 1e6,
            "thousand": 1e3,
            "k": 1e3,
            "percent": 0.01,
            "%": 0.01,
            "": 1.0,
        }
        values = []
        for raw_value, raw_unit in pattern.findall(text):
            value = float(raw_value.replace(",", ""))
            unit = raw_unit.lower()
            if not unit and value.is_integer() and 1900 <= value <= 2099:
                continue
            values.append(value * scales[unit])
        return values

    @classmethod
    def _numeric_conflict(cls, text_a, text_b) -> Optional[float]:
        values_a = cls._quantities(text_a)
        values_b = cls._quantities(text_b)
        if not values_a or not values_b:
            return None
        best_relative_difference = min(
            abs(a - b) / max(abs(a), abs(b), 1.0)
            for a in values_a
            for b in values_b
        )
        return float(np.clip((best_relative_difference - 0.2) / 0.8, 0.0, 1.0))

    def _nli_contradictions(self, sentence_pairs):
        if not sentence_pairs:
            return []
        if not self.use_nli:
            return [None] * len(sentence_pairs)

        directional_pairs = []
        for left, right in sentence_pairs:
            directional_pairs.extend([(left, right), (right, left)])
        probabilities = []
        batch_size = min(self.nli_batch_size, len(directional_pairs))
        start = 0
        while start < len(directional_pairs):
            batch = directional_pairs[start : start + batch_size]
            try:
                encoded = self.nli_tokenizer(
                    [item[0] for item in batch],
                    [item[1] for item in batch],
                    padding=True,
                    truncation=True,
                    max_length=self.nli_max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                with torch.inference_mode():
                    logits = self.nli_model(**encoded).logits
                    batch_probabilities = torch.softmax(logits, dim=-1)[
                        :, self.nli_contradiction_id
                    ]
                probabilities.extend(batch_probabilities.cpu().tolist())
                start += len(batch)
            except RuntimeError as error:
                is_oom = "out of memory" in str(error).lower()
                if not (self.gpu_batch_auto_reduce and is_oom and batch_size > 1):
                    raise
                batch_size = max(1, batch_size // 2)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.warning("NLI OOM; reducing batch size to %d", batch_size)
        return [
            0.5 * (probabilities[2 * index] + probabilities[2 * index + 1])
            for index in range(len(sentence_pairs))
        ]

    def _logic_score(self, triples_a, triples_b, affinity):
        candidates = self._candidate_pairs(affinity)
        if not candidates:
            return 0.5
        sentence_pairs = [
            (
                triples_a[row].get("sentence", ""),
                triples_b[column].get("sentence", ""),
            )
            for _, row, column in candidates
        ]
        nli_scores = self._nli_contradictions(sentence_pairs)
        pair_risks = []
        for (relevance, row, column), (text_a, text_b), nli_score in zip(
            candidates, sentence_pairs, nli_scores
        ):
            components = [
                nli_score,
                self._negation_conflict(text_a, text_b),
                self._temporal_conflict(text_a, text_b),
                self._numeric_conflict(text_a, text_b),
            ]
            active = [index for index, value in enumerate(components) if value is not None]
            active_weights = self.logic_weights[active]
            active_weights = active_weights / active_weights.sum()
            risk = sum(
                float(weight) * float(components[index])
                for weight, index in zip(active_weights, active)
            )
            pair_risks.append((row, column, relevance * risk))

        aligned_rows = sorted({row for row, _, _ in pair_risks})
        aligned_columns = sorted({column for _, column, _ in pair_risks})
        directional_a = [
            max(risk for row, _, risk in pair_risks if row == index)
            for index in aligned_rows
        ]
        directional_b = [
            max(risk for _, column, risk in pair_risks if column == index)
            for index in aligned_columns
        ]
        conflict_risk = 0.5 * (np.mean(directional_a) + np.mean(directional_b))
        comparable_target = min(
            self.coverage_target, max(1, min(len(triples_a), len(triples_b)))
        )
        coverage = min(1.0, len(candidates) / comparable_target)
        return float(np.clip(0.5 + coverage * (0.5 - conflict_risk), 0.0, 1.0))

    @staticmethod
    def _split_sentences(text):
        return [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?。！？])\s*", str(text).strip())
            if sentence.strip()
        ]

    def local_consistency_matrix(self, views):
        sentence_lists = [
            self._split_sentences(view)[: self.MAX_SENTS_PER_VIEW] for view in views
        ]
        flat_sentences = [sentence for sentences in sentence_lists for sentence in sentences]
        if not flat_sentences:
            return np.eye(3, dtype=np.float32)
        embeddings = self._encode(flat_sentences)
        offsets = []
        start = 0
        for sentences in sentence_lists:
            offsets.append((start, start + len(sentences)))
            start += len(sentences)

        local = np.eye(3, dtype=np.float32)
        for i in range(3):
            for j in range(i + 1, 3):
                start_i, end_i = offsets[i]
                start_j, end_j = offsets[j]
                if start_i == end_i or start_j == end_j:
                    score = 0.0
                else:
                    similarity = embeddings[start_i:end_i] @ embeddings[start_j:end_j].T
                    similarity = ((similarity + 1.0) / 2.0).clamp(0.0, 1.0)
                    score = 0.5 * (
                        similarity.max(dim=1).values.mean()
                        + similarity.max(dim=0).values.mean()
                    )
                    score = float(score.item())
                local[i, j] = local[j, i] = score
        return local

    def _encode_sample_features(self, views, triples_per_view):
        """Encode views, triple components, and local sentences in one GPU call."""
        texts = list(views)
        view_indices = list(range(len(views)))
        component_indices = {}
        for view_index, triples in enumerate(triples_per_view):
            for triple_index, triple in enumerate(triples):
                for component in ("subject", "relation", "object"):
                    component_indices[(view_index, triple_index, component)] = len(texts)
                    texts.append(
                        self._normalize_text(triple.get(component, "")) or "[EMPTY]"
                    )

        sentence_lists = [
            self._split_sentences(view)[: self.MAX_SENTS_PER_VIEW] for view in views
        ]
        sentence_indices = []
        for sentences in sentence_lists:
            indices = list(range(len(texts), len(texts) + len(sentences)))
            sentence_indices.append(indices)
            texts.extend(sentences)

        embeddings = self._encode(texts)
        view_embeddings = embeddings[view_indices]
        semantic = ((view_embeddings @ view_embeddings.T + 1.0) / 2.0).clamp(0.0, 1.0)

        local = np.eye(3, dtype=np.float32)
        for i in range(3):
            for j in range(i + 1, 3):
                if not sentence_indices[i] or not sentence_indices[j]:
                    score = 0.0
                else:
                    similarity = embeddings[sentence_indices[i]] @ embeddings[
                        sentence_indices[j]
                    ].T
                    similarity = ((similarity + 1.0) / 2.0).clamp(0.0, 1.0)
                    score = float(
                        (
                            similarity.max(dim=1).values.mean()
                            + similarity.max(dim=0).values.mean()
                        ).item()
                        / 2.0
                    )
                local[i, j] = local[j, i] = score
        return (
            semantic.cpu().numpy().astype(np.float32),
            embeddings,
            component_indices,
            local,
        )

    def close(self):
        if self._openie_pool is not None:
            self._openie_pool.shutdown(wait=True)
            self._openie_pool = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def build_mvc(self, title: str, text: str, desc: str) -> Tuple[np.ndarray, np.ndarray]:
        views = [str(title), str(text), str(desc)]
        if self._openie_pool is None:
            triples_per_view = [self.extract_openie_triples(view) for view in views]
        else:
            triples_per_view = list(
                self._openie_pool.map(self.extract_openie_triples, views)
            )
        semantic, embeddings, indices, local = self._encode_sample_features(
            views, triples_per_view
        )

        factual = np.eye(3, dtype=np.float32)
        logical = np.eye(3, dtype=np.float32)
        for i in range(3):
            for j in range(i + 1, 3):
                affinity = self._affinity_matrix(
                    i, j, triples_per_view, embeddings, indices
                )
                factual[i, j] = factual[j, i] = self._bidirectional_best_alignment(
                    affinity
                )
                logical[i, j] = logical[j, i] = self._logic_score(
                    triples_per_view[i], triples_per_view[j], affinity
                )

        mvc_tensor = np.stack([semantic, factual, logical, local], axis=-1)
        mvc_tensor = np.clip(mvc_tensor, 0.0, 1.0).astype(np.float32)
        combined = (mvc_tensor * self.mvc_weights.reshape(1, 1, 4)).sum(axis=-1)
        return mvc_tensor, combined.astype(np.float32)

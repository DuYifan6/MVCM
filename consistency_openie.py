import re
import logging
from typing import List, Dict, Tuple
import os
import numpy as np
import torch
from pycorenlp import StanfordCoreNLP
from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger("ConsistencyBuilder")


class ConsistencyBuilderOpenIE:
    """
    构建多视图一致性矩阵 MVCM：
      - 语义一致性（semantic）
      - 事实一致性（fact）
      - 逻辑一致性（logic）
      - 局部一致性（local, 句子级）
    返回 (3,3,4) 的 mvc_tensor 以及一个加权融合后的 comb 矩阵。
    """

    # 一些简单的限制，避免 OpenIE / SBERT 过慢
    MAX_TRIPLES_PER_SENT = 3
    MAX_TRIPLES_PER_VIEW = 30
    MAX_SENTS_PER_VIEW = 8

    def __init__(
        self,
        sbert_model: str = "models/all-mpnet-base-v2",
        stanford_url: str = "http://localhost:9000",
        sem_weight: float = 0.5,
        fact_weight: float = 0.2,
        logic_weight: float = 0.15,
        local_weight: float = 0.15,
    ):
        # 1) device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"[MVC] Initializing SBERT on device: {self.device}")

        # 2) SBERT 编码器
        self.sbert = SentenceTransformer(
            sbert_model,
            device=self.device
        )

        try:
            first_param_device = next(self.sbert[0].parameters()).device
            logger.info(f"[MVC] SBERT first layer device: {first_param_device}")
        except Exception as e:
            logger.warning(f"[MVC] Could not inspect SBERT parameters device: {e}")

        # 3) CoreNLP（CPU）
        self.nlp = StanfordCoreNLP(stanford_url)

        # 4) 四通道权重
        self.sem_weight   = sem_weight
        self.fact_weight  = fact_weight
        self.logic_weight = logic_weight
        self.local_weight = local_weight

    # ------------------------------------------------------------------
    # 小工具：文本归一化 / 数字提取 / 主体相似度 / triple 语义相似
    # ------------------------------------------------------------------
    def _normalize_text(self, s: str) -> str:
        return re.sub(r"\W+", " ", (s or "").lower()).strip()

    def _extract_numbers(self, text: str):
        nums = re.findall(r"\d+\.?\d*", text or "")
        return [float(n) for n in nums] if nums else []

    def _subject_sim(self, s1: str, s2: str) -> float:
        """
        主体粗略相似度，用 Jaccard，看 token overlap。
        """
        s1 = self._normalize_text(s1)
        s2 = self._normalize_text(s2)
        if not s1 or not s2:
            return 0.0
        A, B = set(s1.split()), set(s2.split())
        if not A or not B:
            return 0.0
        return len(A & B) / len(A | B)

    def _triple_semantic_sim(self, a: Dict[str, str], b: Dict[str, str]) -> float:
        """
        用 SBERT 对两个三元组做句向量相似度：
        text = "subject relation object"
        """
        try:
            s1 = self._normalize_text(
                f"{a.get('subject','')} {a.get('relation','')} {a.get('object','')}"
            )
            s2 = self._normalize_text(
                f"{b.get('subject','')} {b.get('relation','')} {b.get('object','')}"
            )
            if not s1 or not s2:
                return 0.0
            emb = self.sbert.encode([s1, s2], convert_to_tensor=True)
            sim = util.cos_sim(emb[0:1], emb[1:2]).item()
            # cos_sim ∈ [-1,1] → [0,1]
            return float(max(0.0, min(1.0, (sim + 1.0) / 2.0)))
        except Exception as e:
            logger.debug(f"[MVC] triple_semantic_sim failed: {e}")
            return 0.0

    # ------------------------------------------------------------------
    #  OpenIE 抽取（带数量限制）
    # ------------------------------------------------------------------
    def extract_openie_triples(self, text: str) -> List[Dict[str, str]]:
        try:
            ann = self.nlp.annotate(
                text,
                properties={
                    "annotators": "openie,ner,tokenize,ssplit",
                    "outputFormat": "json",
                },
            )
            if isinstance(ann, str):
                import json
                ann = json.loads(ann)
        except Exception as e:
            logger.error("[MVC] OpenIE annotate failed: %s", e)
            return []

        triples = []
        for sent in ann.get("sentences", []):
            sent_text = " ".join([t.get("word", "") for t in sent.get("tokens", [])]).strip()
            cnt = 0
            for t in sent.get("openie", []):
                triples.append(
                    {
                        "sentence": sent_text,
                        "subject": t.get("subject", ""),
                        "relation": t.get("relation", ""),
                        "object": t.get("object", ""),
                    }
                )
                cnt += 1
                if cnt >= self.MAX_TRIPLES_PER_SENT:
                    break

            if len(triples) >= self.MAX_TRIPLES_PER_VIEW:
                break

        return triples

    # ------------------------------------------------------------------
    #  语义一致性：view 级 SBERT
    # ------------------------------------------------------------------
    def semantic_matrix(self, views: List[str]) -> np.ndarray:
        if not views:
            return np.zeros((0, 0))

        logger.debug(f"[MVC] Encoding {len(views)} views with SBERT on {self.device}")
        emb = self.sbert.encode(views, convert_to_tensor=True)
        logger.debug(f"[MVC] SBERT embedding tensor device: {emb.device}")
        sim = util.cos_sim(emb, emb).detach().cpu().numpy()
        # [-1,1] → [0,1]
        sim = (sim + 1.0) / 2.0
        return sim

    # ------------------------------------------------------------------
    #  事实一致性：对齐再平均 + 数值约束
    # ------------------------------------------------------------------
    def fact_unit_score(self, a: Dict[str, str], b: Dict[str, str]) -> float:
        """
        单对三元组的一致性分数：
          - lexical: subject / relation / object 的精确匹配 + Jaccard
          - semantic: SBERT 对 "s r o" 的相似度
          - numeric: 数字是否冲突（数量/年份等）
        """
        def norm(s):
            return self._normalize_text(s)

        s1, r1, o1 = norm(a.get("subject", "")), norm(a.get("relation", "")), norm(
            a.get("object", "")
        )
        s2, r2, o2 = norm(b.get("subject", "")), norm(b.get("relation", "")), norm(
            b.get("object", "")
        )

        if not (s1 or r1 or o1) or not (s2 or r2 or o2):
            return 0.0

        def jacc(a, b):
            A, B = set(a.split()), set(b.split())
            if not A or not B:
                return 0.0
            return len(A & B) / len(A | B)

        subj = 0.7 * (1.0 if s1 == s2 and s1 else 0.0) + 0.3 * jacc(s1, s2)
        rel  = 0.7 * (1.0 if r1 == r2 and r1 else 0.0) + 0.3 * jacc(r1, r2)
        obj  = 0.7 * (1.0 if o1 == o2 and o1 else 0.0) + 0.3 * jacc(o1, o2)
        lexical_score = (subj + rel + obj) / 3.0  # [0,1]

        # 语义一致性
        semantic_score = self._triple_semantic_sim(a, b)  # [0,1]

        # 数值一致性
        text_a = f"{a.get('subject','')} {a.get('relation','')} {a.get('object','')}"
        text_b = f"{b.get('subject','')} {b.get('relation','')} {b.get('object','')}"
        nums_a = self._extract_numbers(text_a)
        nums_b = self._extract_numbers(text_b)

        if nums_a and nums_b:
            max_a, max_b = max(nums_a), max(nums_b)
            denom = max(abs(max_a), abs(max_b), 1.0)
            rel_diff = abs(max_a - max_b) / denom
            numeric_score = 1.0 if rel_diff < 0.2 else 0.0
        else:
            numeric_score = 1.0  # 没数字默认不冲突

        alpha_lexical = 0.5
        alpha_sem     = 0.3
        alpha_num     = 0.2

        score = (
            alpha_lexical * lexical_score
            + alpha_sem * semantic_score
            + alpha_num * numeric_score
        )
        return float(max(0.0, min(1.0, score)))

    def fact_pair_score(self, triples_a, triples_b, tau=0.1, topk=3):
        """
        Soft factual alignment:
        - 对每个 a，不取 max
        - 用 softmax/top-k 平均，避免被单一 triple 拉满
        """
        if not triples_a or not triples_b:
            return 0.0

        scores_all = []

        for a in triples_a:
            scores = [self.fact_unit_score(a, b) for b in triples_b]
            if not scores:
                continue

            scores = np.array(scores, dtype=float)

            # Top-k 保留最相关的几个
            if topk is not None and len(scores) > topk:
                scores = np.sort(scores)[-topk:]

            # Softmax 加权（tau 控制“接近 max”程度）
            weights = np.exp(scores / tau)
            weights = weights / (weights.sum() + 1e-12)

            soft_score = float((weights * scores).sum())
            scores_all.append(soft_score)

        if not scores_all:
            return 0.0

        return float(np.mean(scores_all))

    # ------------------------------------------------------------------
    #  逻辑一致性：按“冲突比例”打分
    # ------------------------------------------------------------------
    def logic_pair_score(self, triples_a, triples_b):
        """
        Continuous logic consistency:
        score = 1 - conflict_ratio
        """
        years_pat = re.compile(r"\b(19[0-9]{2}|20[0-9]{2})\b")

        if not triples_a or not triples_b:
            return 0.5

        total = 0
        conflicts = 0

        for a in triples_a:
            for b in triples_b:
                subj_sim = self._subject_sim(
                    a.get("subject", ""), b.get("subject", "")
                )
                if subj_sim < 0.4:
                    continue

                full_a = " ".join([a.get("subject", ""), a.get("relation", ""),
                                   a.get("object", ""), a.get("sentence", "")])
                full_b = " ".join([b.get("subject", ""), b.get("relation", ""),
                                   b.get("object", ""), b.get("sentence", "")])

                years_a = set(years_pat.findall(full_a))
                years_b = set(years_pat.findall(full_b))
                nums_a = self._extract_numbers(full_a)
                nums_b = self._extract_numbers(full_b)

                # 只要有可比信息，就计入
                if not (years_a or years_b or nums_a or nums_b):
                    continue

                total += 1

                year_conflict = years_a and years_b and years_a.isdisjoint(years_b)
                num_conflict = False
                if nums_a and nums_b:
                    max_a, max_b = max(nums_a), max(nums_b)
                    diff = abs(max_a - max_b) / max(abs(max_a), abs(max_b), 1.0)
                    if diff > 0.5:
                        num_conflict = True

                if year_conflict or num_conflict:
                    conflicts += 1

        if total == 0:
            return 0.5

        return float(max(0.0, 1.0 - conflicts / total))

    # ------------------------------------------------------------------
    #  局部一致性（Local）：句子级对齐
    # ------------------------------------------------------------------
    def _split_sentences(self, text: str) -> List[str]:
        parts = re.split(r"(?<=[。！？!?\.])\s*", text.strip())
        return [p for p in parts if p]

    def local_consistency_matrix(self, views: List[str]) -> np.ndarray:
        """
        Local consistency: 针对每个视角，将文本切成句子，
        用 SBERT 编码句向量，对不同视角之间的句子集合做最大相似度匹配。
        返回 3x3 矩阵。
        """
        sent_lists = []
        for v in views:
            sents = self._split_sentences(v)
            if len(sents) > self.MAX_SENTS_PER_VIEW:
                sents = sents[: self.MAX_SENTS_PER_VIEW]
            sent_lists.append(sents)

        sent_embs = []
        for sents in sent_lists:
            if not sents:
                sent_embs.append(None)
            else:
                emb = self.sbert.encode(sents, convert_to_tensor=True)
                sent_embs.append(emb)

        N = len(views)
        local = np.zeros((N, N), dtype=float)

        for i in range(N):
            for j in range(N):
                if sent_embs[i] is None or sent_embs[j] is None:
                    local[i, j] = 0.0 if i != j else 1.0
                    continue

                sim = util.cos_sim(sent_embs[i], sent_embs[j])  # (len_i, len_j)
                sim_ij = sim.detach().cpu().numpy()
                sim_ij = (sim_ij + 1.0) / 2.0  # [-1,1] → [0,1]

                row_max = sim_ij.max(axis=1)
                col_max = sim_ij.max(axis=0)
                score = (row_max.mean() + col_max.mean()) / 2.0
                local[i, j] = float(score)

        for k in range(N):
            if local[k, k] == 0.0:
                local[k, k] = 1.0
        return local

    # ------------------------------------------------------------------
    #  综合 MVC 构建
    # ------------------------------------------------------------------
    def build_mvc(self, title: str, text: str, desc: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        返回:
          mvc_tensor: (3,3,4) [semantic, fact, logic, local]
          comb:       (3,3)   4 通道加权融合后的矩阵（0~1 归一化）
        """
        views = [title, text, desc]
        logger.debug("[MVC] Building MVCM (global+local) for one sample")

        N = 3

        # 1) 全局语义一致性（必算）
        sem = self.semantic_matrix(views)  # (3,3)

        # 2) 基于 OpenIE 的事实 & 逻辑
        triples_per_view = [self.extract_openie_triples(v) for v in views]

        lens = [len(t) for t in triples_per_view]  # 每个视角三元组数量

        # coverage: 两个视角都要有一定数量的 triples 才靠谱
        def cov(i, j):
            return float(min(lens[i], lens[j]) / 10.0)  # 10 可调：>=10 视为覆盖充分

        fact = np.zeros((N, N), dtype=float)
        logic = np.ones((N, N), dtype=float) * 0.5  # 默认中性

        for i in range(N):
            for j in range(N):
                if i == j:
                    fact[i, j] = 1.0
                    logic[i, j] = 1.0
                else:
                    c = cov(i, j)
                    fact[i, j] = c * self.fact_pair_score(triples_per_view[i], triples_per_view[j])
                    logic[i, j] = 0.5 + c * (self.logic_pair_score(triples_per_view[i], triples_per_view[j]) - 0.5)

        # 3) 局部一致性（句子级）
        local = self.local_consistency_matrix(views)  # (3,3)

        # 4) 组成 MVCM: (3,3,4)
        mvc_tensor = np.stack([sem, fact, logic, local], axis=-1)

        # 5) 加权融合 + 归一化 → comb
        comb = (
            self.sem_weight * sem
            + self.fact_weight * fact
            + self.logic_weight * logic
            + self.local_weight * local
        )
        mn, mx = comb.min(), comb.max()
        if mx - mn > 1e-12:
            comb = (comb - mn) / (mx - mn)

        return mvc_tensor, comb

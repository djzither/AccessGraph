from __future__ import annotations

from typing import Sequence


class TitleEmbedMatcher:
    def __init__(
        self,
        model_name: str = "intfloat/e5-small-v2",
        threshold: float = 0.78,
    ):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.threshold = threshold

    @staticmethod
    def _prep(text: str) -> str:
        return f"query: {str(text).strip().lower()}"

    def best_match(
        self,
        query_title: str,
        candidate_titles: Sequence[str],
    ) -> tuple[str | None, float]:
        candidates = [str(item) for item in candidate_titles if str(item).strip()]
        if not candidates:
            return None, 0.0

        query_vec = self.model.encode([self._prep(query_title)], normalize_embeddings=True)[0]
        candidate_vecs = self.model.encode(
            [self._prep(item) for item in candidates],
            normalize_embeddings=True,
        )

        scores = candidate_vecs @ query_vec
        best_index = int(scores.argmax())
        best_score = float(scores[best_index])

        if best_score >= self.threshold:
            return candidates[best_index], best_score

        return None, best_score

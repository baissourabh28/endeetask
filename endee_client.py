"""
Endee vector database client.
Wraps the Endee HTTP API for index management, vector upsert, and search.

Key API notes (from source):
- INSERT: POST a top-level JSON array of objects (not wrapped in {"vectors":...})
  meta and filter fields must be JSON strings
- SEARCH: returns application/msgpack — list of [similarity, id, meta, filter, norm, vector] tuples
  meta is raw UTF-8 bytes of the JSON string stored at insert time
- SEARCH filter: must be a JSON-encoded string of an array e.g. '[{"field":{"$eq":"val"}}]'
"""

import json
import requests
import msgpack
from typing import Optional


class EndeeClient:
    def __init__(self, base_url: str, auth_token: str = ""):
        self.base_url = base_url.rstrip("/")
        self.headers = {}
        if auth_token:
            self.headers["Authorization"] = auth_token

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def health(self) -> bool:
        try:
            r = requests.get(self._url("/api/v1/health"), headers=self.headers, timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def create_index(
        self,
        index_name: str,
        dim: int,
        space_type: str = "cosine",
        precision: str = "float32",
        m: int = 16,
        ef_con: int = 128,
    ) -> dict:
        payload = {
            "index_name": index_name,
            "dim": dim,
            "space_type": space_type,
            "precision": precision,
            "M": m,
            "ef_con": ef_con,
        }
        r = requests.post(
            self._url("/api/v1/index/create"),
            json=payload,
            headers={**self.headers, "Content-Type": "application/json"},
        )
        return {"status": r.status_code, "body": r.text}

    def index_info(self, index_name: str) -> dict:
        r = requests.get(self._url(f"/api/v1/index/{index_name}/info"), headers=self.headers)
        if r.status_code == 200:
            return r.json()
        return {}

    def insert_vectors(self, index_name: str, vectors: list[dict]) -> dict:
        """
        vectors: list of dicts with keys:
          - id (str): external document ID
          - vector (list[float]): embedding
          - metadata (dict, optional): serialized to JSON string in 'meta' field
          - filter (dict, optional): filterable fields, serialized to JSON string

        Endee expects a top-level JSON array. meta and filter are JSON strings.
        """
        payload = []
        for v in vectors:
            item: dict = {"id": v["id"], "vector": v["vector"]}
            if "metadata" in v and v["metadata"]:
                # meta must be a JSON string (Endee reads it with .s())
                item["meta"] = json.dumps(v["metadata"])
            if "filter" in v and v["filter"]:
                item["filter"] = json.dumps(v["filter"])
            payload.append(item)

        r = requests.post(
            self._url(f"/api/v1/index/{index_name}/vector/insert"),
            data=json.dumps(payload),
            headers={**self.headers, "Content-Type": "application/json"},
        )
        return {"status": r.status_code, "body": r.text}

    def search(
        self,
        index_name: str,
        vector: list[float],
        k: int = 5,
        filters: Optional[list] = None,
        ef: int = 128,
    ) -> list[dict]:
        """
        Returns list of {id, score, metadata} dicts sorted by score descending.
        filters: Endee filter array e.g. [{"field": {"$eq": "value"}}]
        Response is msgpack: ResultSet {results: [{similarity, id, meta, ...}]}
        """
        payload: dict = {"vector": vector, "k": k, "ef": ef}
        if filters:
            payload["filter"] = json.dumps(filters)

        r = requests.post(
            self._url(f"/api/v1/index/{index_name}/search"),
            data=json.dumps(payload),
            headers={**self.headers, "Content-Type": "application/json"},
        )
        if r.status_code != 200:
            return []

        try:
            data = msgpack.unpackb(r.content, raw=False)
            results = []
            # Response is a list of [similarity, id, meta, filter, norm, vector] tuples
            for item in data:
                meta = {}
                if isinstance(item, (list, tuple)) and len(item) >= 3:
                    similarity = item[0]
                    vec_id = item[1]
                    raw_meta = item[2]
                else:
                    continue
                if raw_meta:
                    try:
                        if isinstance(raw_meta, (bytes, bytearray)):
                            meta = json.loads(raw_meta.decode("utf-8"))
                        elif isinstance(raw_meta, str):
                            meta = json.loads(raw_meta)
                    except Exception:
                        pass
                results.append({
                    "id": vec_id,
                    "score": similarity,
                    "metadata": meta,
                })
            return results
        except Exception:
            return []

    def delete_index(self, index_name: str) -> bool:
        r = requests.delete(
            self._url(f"/api/v1/index/{index_name}/delete"),
            headers=self.headers,
        )
        return r.status_code in (200, 204)

    def delete_vector(self, index_name: str, vector_id: str) -> bool:
        r = requests.delete(
            self._url(f"/api/v1/index/{index_name}/vector/{vector_id}/delete"),
            headers=self.headers,
        )
        return r.status_code in (200, 204)

    def list_indexes(self) -> list[str]:
        r = requests.get(self._url("/api/v1/index/list"), headers=self.headers)
        if r.status_code == 200:
            data = r.json()
            # Returns list of index objects with 'name' field
            indexes = data.get("indexes", [])
            if indexes and isinstance(indexes[0], dict):
                return [f"endee/{idx['name']}" for idx in indexes]
            return indexes
        return []

"""On-disk cluster metadata (``metadata.json``) management.

:class:`MetadataStore` owns the ``metadata.json`` file for each cluster type.
It is responsible for nothing beyond reading, writing, and querying that file.
All business logic (centroid computation, similarity search) lives elsewhere.

File format (per type)
----------------------
``{base_path}/{cluster_type}_custom/metadata.json``::

    {
        "ClusterName": {
            "class": "sdk_issue",
            "run_ids": {
                "QNN-v2.46.0.260319041023_nightly": {
                    "tc-uuid-1234": {
                        "runtime": "htp_fp16",
                        "soc_name": "Kailua",
                        "BU": "WLAN",
                        ...
                    }
                }
            }
        }
    }

**Ordering invariant**: the key order of the dict returned by :meth:`load`
must always match the row order of the centroid matrix in
:class:`~src.clustering.vector_store.VectorStore`.

Layering
--------
Imports only from ``src.core``, ``src.constants``, ``src.logger``,
and the standard library.
"""

from __future__ import annotations

import json
import os
from typing import Optional

from src.constants import FaissConfigurations
from src.core.exceptions import VectorStoreError
from src.core.interfaces import IMetadataStore
from src.logger import AppLogger

logger = AppLogger().get_logger(__name__)

__all__ = ["MetadataStore"]


class MetadataStore(IMetadataStore):
    """Manages ``metadata.json`` and run-ID tracking files.

    One instance is typically shared for the whole process.  Each cluster type
    has its own sub-directory under *base_path*::

        {base_path}/{cluster_type}_custom/metadata.json
        {base_path}/{cluster_type}_custom/processed_runids.json

    Args:
        base_path: Root directory for the cluster index.  Defaults to
            :attr:`~src.constants.FaissConfigurations.base_path`.

    Example:
        >>> ms = MetadataStore()
        >>> meta = ms.load("quantizer")
        >>> ms.save("quantizer", meta)
        >>> ms.mark_run_processed("quantizer_root_dir", "QNN-v2.46.0...")
    """

    def __init__(self, base_path: str = FaissConfigurations.base_path) -> None:
        self.base_path = base_path

    # ------------------------------------------------------------------
    # IMetadataStore contract
    # ------------------------------------------------------------------

    def load(self, cluster_type: str) -> dict:
        """Load metadata for *cluster_type* from disk.

        Args:
            cluster_type: Test-type identifier (e.g. ``"quantizer"``).

        Returns:
            Metadata dict (empty dict if no file exists yet).

        Raises:
            VectorStoreError: On I/O failure.
        """
        path = self._metadata_path(cluster_type)
        if not os.path.isfile(path):
            return {}
        try:
            with open(path, "r") as fh:
                metadata = json.load(fh)
            logger.debug(f"[MetadataStore] Loaded {len(metadata)} clusters for type={cluster_type}")
            return metadata
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to load metadata for type={cluster_type}", index_path=path, cause=exc
            ) from exc

    def save(self, cluster_type: str, metadata: dict) -> None:
        """Persist metadata for *cluster_type* to disk.

        Creates the type directory if it does not exist.

        Args:
            cluster_type: Test-type identifier.
            metadata: Full metadata dict to write.

        Raises:
            VectorStoreError: On I/O failure.
        """
        type_dir = self._type_dir(cluster_type)
        os.makedirs(type_dir, exist_ok=True)
        path = os.path.join(type_dir, "metadata.json")
        try:
            with open(path, "w") as fh:
                json.dump(metadata, fh, indent=3)
            logger.info(f"[MetadataStore] Saved {len(metadata)} clusters for type={cluster_type}")
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to save metadata for type={cluster_type}", index_path=path, cause=exc
            ) from exc

    def get_cluster_names(self, cluster_type: str) -> list[str]:
        """Return the ordered list of cluster names for *cluster_type*.

        The order matches the centroid row order in the corresponding
        :class:`~src.clustering.vector_store.VectorStore`.

        Args:
            cluster_type: Test-type identifier.

        Returns:
            List of cluster name strings; empty list if no metadata exists.
        """
        return list(self.load(cluster_type).keys())

    # ------------------------------------------------------------------
    # Run-ID tracking
    # ------------------------------------------------------------------

    def is_run_processed(self, cluster_type: str, run_id: str) -> bool:
        """Return ``True`` if *run_id* was previously processed for *cluster_type*.

        Args:
            cluster_type: Test-type identifier.
            run_id: Run ID string to check.

        Returns:
            ``True`` if the run was already processed.
        """
        return run_id in self._load_processed_runids(self._type_dir(cluster_type))

    def mark_run_processed(self, cluster_type: str, run_id: str) -> None:
        """Record *run_id* as processed in ``processed_runids.json``.

        Args:
            cluster_type: Test-type identifier.
            run_id: Run ID to mark as done.
        """
        type_dir = self._type_dir(cluster_type)
        os.makedirs(type_dir, exist_ok=True)
        processed = self._load_processed_runids(type_dir)
        if run_id not in processed:
            processed.append(run_id)
            path = os.path.join(type_dir, "processed_runids.json")
            with open(path, "w") as fh:
                json.dump(processed, fh, indent=3)

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    def update_run_metadata(
        self,
        cluster_type: str,
        cluster_name: str,
        run_id: str,
        run_metadata: dict,
    ) -> None:
        """Add *run_metadata* for *run_id* under *cluster_name* in the metadata.

        Loads the metadata file, inserts the new run_id entry, and saves.
        Does nothing if *run_id* already exists under *cluster_name*.

        Args:
            cluster_type: Test-type identifier.
            cluster_name: Cluster to update.
            run_id: Run ID to add.
            run_metadata: TC UUID → record dict for this run.
        """
        metadata = self.load(cluster_type)
        if cluster_name not in metadata:
            logger.warning(
                f"[MetadataStore] Cluster '{cluster_name}' not found for type={cluster_type}; "
                "skipping run metadata update."
            )
            return
        if run_id not in metadata[cluster_name].get("run_ids", {}):
            metadata[cluster_name].setdefault("run_ids", {})[run_id] = run_metadata
            self.save(cluster_type, metadata)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _type_dir(self, cluster_type: str) -> str:
        return os.path.join(self.base_path, f"{cluster_type}_custom")

    def _metadata_path(self, cluster_type: str) -> str:
        return os.path.join(self._type_dir(cluster_type), "metadata.json")

    @staticmethod
    def _load_processed_runids(type_dir: str) -> list[str]:
        path = os.path.join(type_dir, "processed_runids.json")
        if not os.path.isfile(path):
            return []
        try:
            with open(path, "r") as fh:
                return json.load(fh)
        except Exception:
            logger.warning("[MetadataStore] Failed to load processed_runids.json at %s; treating as empty", path)
            return []

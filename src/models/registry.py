"""
DockInsights — Model Registry.

Tracks trained model versions, their metrics, and artifacts.
Enables reproducibility, metadata validation, and safe model rollback.
"""

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.config import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ModelError(Exception):
    """Base exception for model registry errors."""
    pass


class ModelNotFoundError(ModelError, FileNotFoundError):
    """Raised when a requested model artifact or version does not exist."""
    pass


class ModelMismatchError(ModelError, ValueError):
    """Raised when loaded model metadata does not match requested model identity parameters."""
    pass


class ModelIntegrityError(ModelError, ValueError):
    """Raised when model artifact is corrupted, empty, or cannot be deserialized."""
    pass


class ModelRegistry:
    """
    Local model registry backed by the models/ directory structure.

    Directory structure:
        models/
        └── {model_name}/
            ├── active_version.txt
            ├── {model_version}/
            │   ├── model.pkl
            │   └── metadata.json
            └── {model_version_2}/
                ├── model.pkl
                └── metadata.json
    """

    def __init__(self, base_dir: Optional[str] = None):
        settings = get_settings()
        self.base_dir = Path(base_dir or settings.trained_models_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_model(
        self,
        model: Any,
        model_name: str,
        model_version: str,
        metrics: Dict[str, float],
        hyperparameters: Optional[Dict[str, Any]] = None,
        route: Optional[str] = None,
        origin: Optional[str] = None,
        destination: Optional[str] = None,
        vessel_class: Optional[str] = None,
        cargo_type: Optional[str] = None,
        features: Optional[List[str]] = None,
        validation_methodology: Optional[str] = None,
        is_active: bool = True,
        data_mode: str = "SYNTHETIC_DEMO",
        provenance_status: str = "SYNTHETIC_DEMO",
        dataset_name: Optional[str] = "freight_rates",
        dataset_version: Optional[str] = "v2.0",
    ) -> str:
        """
        Save a trained model with its metadata and register its version.
        """
        version_dir = self.base_dir / model_name / model_version
        version_dir.mkdir(parents=True, exist_ok=True)

        artifact_path = version_dir / "model.pkl"
        metadata_path = version_dir / "metadata.json"

        # Serialize model
        try:
            if hasattr(model, "save") and callable(getattr(model, "save")):
                model.save(str(artifact_path))
            else:
                with open(artifact_path, "wb") as f:
                    pickle.dump(model, f)
        except Exception as e:
            logger.error(f"Failed to serialize model to {artifact_path}: {e}")
            raise ModelIntegrityError(f"Model serialization failed: {e}") from e

        # Construct metadata
        metadata = {
            "training_date": datetime.now(timezone.utc).isoformat(),
            "dataset_name": dataset_name or "freight_rates",
            "dataset_version": dataset_version or "v2.0",
            "data_mode": data_mode or "SYNTHETIC_DEMO",
            "provenance_status": provenance_status or "SYNTHETIC_DEMO",
            "is_verified_external": False,
            "model_name": model_name,
            "model_version": model_version,
            "route": route,
            "origin": origin,
            "destination": destination,
            "vessel_class": vessel_class,
            "cargo_type": cargo_type or "thermal_coal",
            "features": features or getattr(model, "feature_cols", []),
            "metrics": metrics,
            "validation_methodology": validation_methodology or "OOS_chronological_holdout",
            "artifact_path": str(artifact_path),
            "hyperparameters": hyperparameters or {},
        }

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        if is_active:
            self.register_version(model_name, model_version, is_active=True)

        logger.info(f"Successfully saved model '{model_name}' version '{model_version}' to {artifact_path}")
        return str(artifact_path)

    def load_model(
        self,
        model_name: str,
        model_version: Optional[str] = None,
        route: Optional[str] = None,
        vessel_class: Optional[str] = None,
        cargo_type: Optional[str] = None,
    ) -> Any:
        """
        Load a model by name and version with identity validation.

        If model_version is None, loads the currently active version.

        Args:
            model_name: Model identifier.
            model_version: Specific version string, or None for active.
            route: Expected route to validate metadata against.
            vessel_class: Expected vessel class to validate metadata against.
            cargo_type: Expected cargo type to validate metadata against.

        Returns:
            Deserialized model object.

        Raises:
            ModelNotFoundError: If artifact or version does not exist.
            ModelMismatchError: If loaded metadata violates requested parameters.
            ModelIntegrityError: If artifact file is corrupted or empty.
        """
        version = model_version or self.get_active_version(model_name)
        if not version:
            versions = self.list_versions(model_name)
            if versions:
                version = versions[-1]
            else:
                raise ModelNotFoundError(f"No trained models or versions found for '{model_name}' in {self.base_dir}")

        version_dir = self.base_dir / model_name / version
        if not version_dir.exists():
            raise ModelNotFoundError(f"Model version '{version}' for '{model_name}' not found at {version_dir}")

        artifact_path = version_dir / "model.pkl"
        if not artifact_path.exists():
            artifact_joblib = version_dir / "model.joblib"
            if artifact_joblib.exists():
                artifact_path = artifact_joblib
            else:
                raise ModelNotFoundError(f"No model artifact (model.pkl) found in {version_dir}")

        if artifact_path.stat().st_size == 0:
            raise ModelIntegrityError(f"Model artifact at {artifact_path} is empty (0 bytes).")

        # Validate metadata if present
        metadata_path = version_dir / "metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)

                if route and meta.get("route") and meta["route"].upper() != route.upper():
                    raise ModelMismatchError(
                        f"Model identity mismatch for route: requested '{route}', but model was trained for '{meta['route']}'"
                    )

                if vessel_class and meta.get("vessel_class") and meta["vessel_class"].lower() != vessel_class.lower():
                    raise ModelMismatchError(
                        f"Model identity mismatch for vessel_class: requested '{vessel_class}', but model was trained for '{meta['vessel_class']}'"
                    )

                if cargo_type and meta.get("cargo_type") and meta["cargo_type"].lower() != cargo_type.lower():
                    raise ModelMismatchError(
                        f"Model identity mismatch for cargo_type: requested '{cargo_type}', but model was trained for '{meta['cargo_type']}'"
                    )
            except (ModelMismatchError, ModelNotFoundError):
                raise
            except Exception as e:
                logger.warning(f"Could not parse metadata at {metadata_path}: {e}")

        # Deserialize model
        try:
            with open(artifact_path, "rb") as f:
                loaded_obj = pickle.load(f)

            if isinstance(loaded_obj, dict) and not hasattr(loaded_obj, "predict"):
                m_type = (model_name or "").lower()
                if "xgboost" in m_type:
                    from src.models.xgboost_forecaster import XGBoostForecaster
                    model = XGBoostForecaster()
                elif "arima" in m_type or "sarima" in m_type:
                    from src.models.arima_forecaster import ARIMAForecaster
                    model = ARIMAForecaster()
                elif "ensemble" in m_type:
                    from src.models.ensemble_forecaster import EnsembleForecaster
                    model = EnsembleForecaster()
                else:
                    from src.models.baseline_forecaster import BaselineForecaster
                    model = BaselineForecaster()
                model.load(str(artifact_path))
            else:
                model = loaded_obj

            logger.info(f"Successfully loaded model '{model_name}' version '{version}' from {artifact_path}")
            return model
        except (ModelMismatchError, ModelNotFoundError, ModelIntegrityError):
            raise
        except Exception as e:
            raise ModelIntegrityError(f"Failed to deserialize model artifact from {artifact_path}: {e}") from e

    def get_metadata(self, model_name: str, model_version: Optional[str] = None) -> Dict[str, Any]:
        """Fetch metadata dictionary for a model version."""
        version = model_version or self.get_active_version(model_name)
        if not version:
            versions = self.list_versions(model_name)
            version = versions[-1] if versions else None
        if not version:
            return {}
        metadata_path = self.base_dir / model_name / version / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def list_versions(self, model_name: str) -> List[str]:
        """List all available versions of a model sorted chronologically/semantically."""
        model_dir = self.base_dir / model_name
        if not model_dir.exists():
            return []
        return sorted([d.name for d in model_dir.iterdir() if d.is_dir()])

    def get_active_version(self, model_name: str) -> Optional[str]:
        """Get the currently active (production) version string of a model."""
        active_file = self.base_dir / model_name / "active_version.txt"
        if active_file.exists():
            try:
                with open(active_file, "r", encoding="utf-8") as f:
                    return f.read().strip()
            except Exception as e:
                logger.warning(f"Could not read active version for '{model_name}': {e}")
        
        versions = self.list_versions(model_name)
        return versions[-1] if versions else None

    def register_version(self, model_name: str, model_version: str, is_active: bool = True) -> None:
        """Register a model version and optionally mark it as active."""
        model_dir = self.base_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        if is_active:
            active_file = model_dir / "active_version.txt"
            with open(active_file, "w", encoding="utf-8") as f:
                f.write(model_version)
            logger.info(f"Set active version for '{model_name}' to '{model_version}'")

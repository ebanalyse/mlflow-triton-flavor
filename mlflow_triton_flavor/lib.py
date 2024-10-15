import os
import shutil
import sys

from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME, ModelInfo
from mlflow.protos.databricks_pb2 import RESOURCE_ALREADY_EXISTS

FLAVOR_NAME = "triton"


def save_model(source_path: str, path: str, mlflow_model=Model()):
    """
    Save a Triton model to a path on the local file system.

    :param source_path: Path to Triton model folder.
    :param path: Path to where the model should be saved.
    :param mlflow_model: :py:mod:`mlflow.models.Model` this flavor is being added to.

    """

    dest_path = os.path.abspath(path)
    if os.path.exists(dest_path):
        raise MlflowException(
            message="Path '{}' already exists".format(dest_path),
            error_code=RESOURCE_ALREADY_EXISTS,
        )
    os.makedirs(dest_path)

    source_path = os.path.normpath(source_path)
    model_data_subpath = os.path.basename(source_path)
    model_data_path = os.path.join(dest_path, model_data_subpath)

    shutil.copytree(source_path, model_data_path)

    mlflow_model.add_flavor(FLAVOR_NAME, data=model_data_subpath)
    mlflow_model.save(os.path.join(dest_path, MLMODEL_FILE_NAME))


def log_model(path: str, register_model: bool = True) -> ModelInfo:
    """
    Log a Triton model as an MLflow artifact for the current run.

    :param path: Path to Triton model folder.
    :param register_model: If given, create a model version, also creating a registered
                            model if one with the given name does not exist. The name is
                            equal to the basename of the ``path``.

    """
    model_name = os.path.basename(os.path.normpath(path))
    return Model.log(
        artifact_path=model_name,
        flavor=sys.modules[__name__],
        source_path=path,
        registered_model_name=model_name if register_model else None,
    )

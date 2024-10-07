import os
import shutil
import sys

from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.protos.databricks_pb2 import RESOURCE_ALREADY_EXISTS
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS

FLAVOR_NAME = "triton"


def save_model(triton_model_path, path, mlflow_model=None):
    """
    Save a Triton model to a path on the local file system.

    :param triton_model_path: File path to Triton model to be saved.
    :param path: Local path where the model is to be saved.
    :param mlflow_model: :py:mod:`mlflow.models.Model` this flavor is being added to.

    """

    path = os.path.abspath(path)
    if os.path.exists(path):
        raise MlflowException(
            message="Path '{}' already exists".format(path),
            error_code=RESOURCE_ALREADY_EXISTS,
        )
    os.makedirs(path)
    triton_model_path = os.path.normpath(triton_model_path)
    model_data_subpath = os.path.basename(triton_model_path)
    model_data_path = os.path.join(path, model_data_subpath)

    # Save Triton model
    shutil.copytree(triton_model_path, model_data_path)

    mlflow_model.add_flavor(FLAVOR_NAME, data=model_data_subpath)
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))


def log_model(
    triton_model_path,
    artifact_path,
    registered_model_name=None,
    await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
):
    """
    Log a Triton model as an MLflow artifact for the current run.

    :param triton_model_path: File path to Triton model.
    :param artifact_path: Run-relative artifact path.
    :param registered_model_name: (Experimental) If given, create a model version under
                                  ``registered_model_name``, also creating a registered model if one
                                  with the given name does not exist.

    :param await_registration_for: Number of seconds to wait for the model version to finish
                            being created and is in ``READY`` status. By default, the function
                            waits for five minutes. Specify 0 or None to skip waiting.

    """
    Model.log(
        artifact_path=artifact_path,
        flavor=sys.modules[__name__],
        triton_model_path=triton_model_path,
        registered_model_name=registered_model_name,
        await_registration_for=await_registration_for,
    )

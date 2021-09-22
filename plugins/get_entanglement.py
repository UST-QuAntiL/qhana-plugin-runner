# Copyright 2021 QHAna plugin runner contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
from http import HTTPStatus
from json import dumps, loads
from tempfile import SpooledTemporaryFile
from typing import Mapping, Optional, List, Dict

import marshmallow as ma
from celery.canvas import chain
from celery.result import AsyncResult
from celery.utils.log import get_task_logger
from flask import Response
from flask import redirect
from flask.app import Flask
from flask.globals import request
from flask.helpers import url_for
from flask.templating import render_template
from flask.views import MethodView
from marshmallow import EXCLUDE
from sqlalchemy.sql.expression import select

from qhana_plugin_runner.db.db import DB
from qhana_plugin_runner.api.plugin_schemas import PluginMetadataSchema
from qhana_plugin_runner.api.util import (
    FrontendFormBaseSchema,
    MaBaseSchema,
    SecurityBlueprint,
    FileUrl,
)
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask

from qhana_plugin_runner.requests import open_url
from qhana_plugin_runner.storage import STORE
from qhana_plugin_runner.tasks import save_task_error, save_task_result
from qhana_plugin_runner.util.plugins import QHAnaPluginBase, plugin_identifier

_plugin_name = "get-Entanglement"
__version__ = "v0.1.0"
_identifier = plugin_identifier(_plugin_name, __version__)

GetEntanglement_BLP = SecurityBlueprint(
    _identifier,  # blueprint name
    __name__,  # module import name!
    description="Get the Schmidt Rank of a given quatum state as a measurement of it's entanglement.",
)


class TaskResponseSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_id = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_result_url = ma.fields.Url(required=True, allow_none=False, dump_only=True)



class InputParametersSchema(FrontendFormBaseSchema):
    input_data = FileUrl(
        required=True, allow_none=False, load_only=True, metadata={"label": "Input Data"}
    )


@GetEntanglement_BLP.route("/")
class PluginsView(MethodView):
    """Plugins collection resource."""

    @GetEntanglement_BLP.response(HTTPStatus.OK, PluginMetadataSchema)
    @GetEntanglement_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Get Entanglement endpoint returning the plugin metadata."""
        return {
            "name": GetEntanglement_BLP.instance.name,
            "version": GetEntanglement_BLP.instance.version,
            "identifier": GetEntanglement_BLP.instance.identifier,
            "root_href": url_for(f"{GetEntanglement_BLP.name}.PluginsView"),
            "title": "Get Entanglement of given state",
            "description": "Computes Schmidt rank as a measure of entanglement.",
            "plugin_type": "entanglement-calculation",
            "tags": [],
            "processing_resource_metadata": {
                "href": url_for(f"{GetEntanglement_BLP.name}.CalcSimilarityView"),
                "ui_href": url_for(f"{GetEntanglement_BLP.name}.MicroFrontend"),
                "inputs": [
                    [
                        {
                            "output_type": "state vector",
                            "content_type": "application/json",
                            "name": "state vector which should have len 2^k (k = #qubits) and be normalized",
                        },
                    ]
                ],
                "outputs": [
                    [
                        {
                            "output_type": "Schmidt rank",
                            "content_type": "application/zip",
                            "name": "Schmidt rank of the given state",
                        }
                    ]
                ],
            },
        }

@GetEntanglement_BLP.route("/ui/")
class MicroFrontend(MethodView):
    """Micro frontend for the get entanglement plugin."""

    example_inputs = {
        "inputStr": "Sample input string.",
    }

    @GetEntanglement_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the get entanglement plugin."
    )
    @GetEntanglement_BLP.arguments(
        InputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @GetEntanglement_BLP.require_jwt("jwt", optional=True)
    def get(self, errors):
        """Return the micro frontend."""
        return self.render(request.args, errors)

    @GetEntanglement_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the get entanglement plugin."
    )
    @GetEntanglement_BLP.arguments(
        InputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @GetEntanglement_BLP.require_jwt("jwt", optional=True)
    def post(self, errors):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors)

    def render(self, data: Mapping, errors: dict):
        schema = InputParametersSchema()
        return Response(
            render_template(
                "simple_template.html",
                name=GetEntanglementPlugin.instance.name,
                version=GetEntanglementPlugin.instance.version,
                schema=schema,
                values=data,
                errors=errors,
                process=url_for(f"{GetEntanglement_BLP.name}.CalcSimilarityView"),
                example_values=url_for(
                    f"{GetEntanglement_BLP.name}.MicroFrontend", **self.example_inputs
                ),
            )
        )


@GetEntanglement_BLP.route("/process/")
class GetEntanglementView(MethodView):
    """Start a long running processing task."""

    @GetEntanglement_BLP.arguments(InputParametersSchema(unknown=EXCLUDE), location="form")
    @GetEntanglement_BLP.response(HTTPStatus.OK, TaskResponseSchema())
    @GetEntanglement_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments):
        """Start the calculation task."""
        db_task = ProcessingTask(
            task_name=calculation_task.name, parameters=dumps(arguments)
        )
        db_task.save(commit=True)

        # all tasks need to know about db id to load the db entry
        task: chain = calculation_task.s(db_id=db_task.id) | save_task_result.s(
            db_id=db_task.id
        )
        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        result: AsyncResult = task.apply_async()

        db_task.task_id = result.id
        db_task.save(commit=True)

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(result.id)), HTTPStatus.SEE_OTHER
        )


class GetEntanglementPlugin(QHAnaPluginBase):

    name = _plugin_name
    version = __version__

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

    def get_api_blueprint(self):
        return GetEntanglement_BLP

    def get_requirements(self) -> str:
        return "numpy >= 1.8.0"

TASK_LOGGER = get_task_logger(__name__)


@CELERY.task(name=f"{GetEntanglementPlugin.instance.identifier}.get_entanglement_task", bind=True,)
def calculation_task(self, db_id: int) -> str:
    import numpy as np

    TASK_LOGGER.info(f"Starting new get entanglement task with db id '{db_id}'")
    
    
    task_data: ProcessingTask = DB.session.execute(select(ProcessingTask).filter_by(id=db_id)).scalar_one()

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    params: Dict = loads(task_data.parameters or "{}")
    input_data_url: str = params.get("input_data", None)
    


    TASK_LOGGER.info(
        f"input_data: {input_data_url}"
    )

    if None in [input_data_url]:
        raise ValueError("Request is missing one or more values.")

    with open_url(input_data_url, stream=True) as url_data:
        input_data_arr = np.genfromtxt(url_data.iter_lines(), delimiter=",")


    #Norm coeff vector sice a state vector is normed
    coeff = input_data_arr/np.linalg.norm(input_data_arr) 
    
    #Check for correct state vector --> len() must be 2^k
    n_huge = int(np.log2(len(coeff)))
    if len(coeff) != 2**n_huge:
        raise ValueError("Given Coefficient Vector does not have length 2^k")

    #Calculate Beta Matrix out of Coefficient Vector
    n = int(np.ceil(n_huge/2))  
    m = n_huge-n
    beta = np.zeros((2**n, 2**m), dtype=np.cdouble)
    formatstring = "0" + str(int(np.log2(len(coeff)))) + "b"
    x_labels = []

    for i in range(len(coeff)):
        x_labels.append(format(i, formatstring))

    for i in range(len(coeff)):
    	k = int(x_labels[i][:n], 2)
    	l = int(x_labels[i][-m:], 2)
    	beta[k, l] = coeff[i]
        
    #Do SVD    
    u_, a, v_ = np.linalg.svd(beta)

    #Get Schmidt Rank as number of non zero entries in a
    zero_entries = np.isclose(a, np.zeros_like(a))
    grade = np.count_nonzero(zero_entries == 0) 

    #Save Schmidt Rank
    with SpooledTemporaryFile(mode="w") as output:
        np.savetxt(output, grade, delimiter=",")
        STORE.persist_task_result(
            db_id, output, "out.csv", "get_entanglement-result", "text/csv"
        )
        
    return "Result stored in file"

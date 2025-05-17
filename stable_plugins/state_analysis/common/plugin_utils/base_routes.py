import json
from http import HTTPStatus

from flask import Response, abort, redirect, render_template, request, url_for
from flask.views import MethodView
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.tasks import save_task_error, save_task_result


class BaseMicroFrontend(MethodView):

    Plugin = None
    SchemaClass = None
    help_text = ""
    example_inputs = {}

    def render_view(self, data, errors, valid):
        plugin = self.Plugin.instance
        if plugin is None:
            abort(HTTPStatus.INTERNAL_SERVER_ERROR)
        schema = self.SchemaClass()
        result = None
        task_id = data.get("task_id")
        if task_id:
            task = ProcessingTask.get_by_id(task_id)
            if task:
                result = task.result
        return Response(
            render_template(
                "simple_template.html",
                name=plugin.name,
                version=plugin.version,
                schema=schema,
                valid=valid,
                values=data,
                errors=errors,
                result=result,
                process=url_for(f"{plugin.identifier}.ProcessView"),
                help_text=self.help_text,
                example_values=url_for(
                    f"{plugin.identifier}.MicroFrontend", **self.example_inputs
                ),
            )
        )

    def get(self, errors):
        return self.render_view(request.args, errors, valid=False)

    def post(self, errors):
        return self.render_view(request.form, errors, valid=(not errors))


class BaseProcessView(MethodView):
    task_function = None

    def post(self, arguments):
        db_task = ProcessingTask(
            task_name=self.task_function.name, parameters=json.dumps(arguments)
        )
        db_task.save(commit=True)
        task_chain = self.task_function.s(db_id=db_task.id) | save_task_result.s(
            db_id=db_task.id
        )
        task_chain.link_error(save_task_error.s(db_id=db_task.id))
        task_chain.apply_async()
        db_task.save(commit=True)
        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)),
            code=HTTPStatus.SEE_OTHER,
        )


class BasePluginView(MethodView):
    """
    Stellt eine generische Plugin-Metadaten-View bereit.
    In den Unterklassen müssen vor allem das Plugin-Objekt und
    gegebenenfalls plugin-spezifische Anpassungen (z.B. data_output) gesetzt werden.
    """

    Plugin = None  # Muss in der Unterklasse gesetzt werden
    data_output = []  # Optional: Ausgabe-Daten, je nach Plugin

    def get(self):
        plugin = self.Plugin.instance
        if plugin is None:
            abort(HTTPStatus.INTERNAL_SERVER_ERROR)

        return PluginMetadata(
            title=plugin.name,
            description=plugin.description,
            name=plugin.name,
            version=plugin.version,
            type=PluginType.processing,
            entry_point=EntryPoint(
                href=url_for(f"{plugin.identifier}.ProcessView"),
                ui_href=url_for(f"{plugin.identifier}.MicroFrontend"),
                plugin_dependencies=[],  # ggf. anpassen
                data_input=[
                    DataMetadata(
                        data_type="application/json",
                        content_type=["application/json"],
                        required=True,
                    )
                ],
                data_output=self.data_output,
            ),
            tags=plugin.tags,
        )

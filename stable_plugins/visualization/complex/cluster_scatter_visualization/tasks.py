# Copyright 2022 QHAna plugin runner contributors.
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

from json import dumps
from pathlib import Path
from tempfile import SpooledTemporaryFile
from typing import Any, Dict, Optional, Set, Tuple, Union

import muid
import pandas as pd
import plotly.express as px
from celery.utils.log import get_task_logger
from requests import HTTPError

from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.virtual_plugins import DataBlob, PluginState
from qhana_plugin_runner.plugin_utils.entity_marshalling import (
    ensure_array,
    ensure_dict,
    load_entities,
)
from qhana_plugin_runner.requests import get_mimetype, open_url, retrieve_filename
from qhana_plugin_runner.storage import STORE

from . import ClusterScatterVisualization

TASK_LOGGER = get_task_logger(__name__)


class PlotNotFinishedError(Exception):
    pass


def get_readable_hash(s: str) -> str:
    return muid.pretty(muid.bhash(s.encode("utf-8")), k1=6, k2=5).replace(" ", "-")


def _get_plot(
    entity_url: str,
    clusters_url: Optional[str],
    entity_data_url: Optional[str],
    full_html: bool,
) -> Tuple[str, str]:
    """Generate a scatter plot from the given url.

    Args:
        entity_url (str): the url containing entity coordinates
        clusters_url (str|None): an optional url containing entity cluster labels
        entity_data_url (str|None): an optional url containing original entity attributes
        full_html (bool): if True, produce a standalone html page, else produce
            an embeddable html snippet.

    Returns:
        Tuple[str, str]: html_content, filename
    """
    is_3d = False
    entities: Dict[str, Dict[str, Any]] = {}
    with open_url(entity_url) as response:
        mimetype = get_mimetype(response)
        if mimetype is None:
            raise ValueError("Could not determine mimetype.")
        name = retrieve_filename(response)
        for ent in ensure_array(load_entities(response, mimetype=mimetype)):
            entity_id = str(ent.ID)
            diagram_ent: Dict[str, Any] = {
                "ID": entity_id,
                "name": entity_id,
            }
            dim = len(ent.values)
            if dim == 0:
                raise ValueError("Cannot produce scatter plot without coordinates.")
            diagram_ent["x"] = ent.values[0]
            if dim == 1:
                diagram_ent["y"] = 0
            if dim > 1:
                diagram_ent["y"] = ent.values[1]
            if dim > 2:
                diagram_ent["z"] = ent.values[2]
                is_3d = True
            diagram_ent["href"] = ent.href
            diagram_ent["attributes"] = {}
            entities[entity_id] = diagram_ent

    if entity_data_url:
        with open_url(entity_data_url) as response:
            mimetype = get_mimetype(response)
            if mimetype is None:
                raise ValueError("Could not determine mimetype.")
            for ent in ensure_dict(load_entities(response, mimetype=mimetype)):
                entity_id = ent.get("ID")
                if entity_id is None:
                    continue
                diag_ent = entities.get(str(entity_id), None)
                if diag_ent is None:
                    continue
                href = ent.get("href")
                if href:
                    diag_ent["href"] = href
                extra_data: Dict[str, Union[str, float, int, bool, None]] = {}
                for key, value in ent.items():
                    if key in {"ID", "href"}:
                        continue
                    if value is None or isinstance(value, (str, float, int, bool)):
                        extra_data[str(key)] = value
                    else:
                        try:
                            extra_data[str(key)] = dumps(value, sort_keys=True)
                        except TypeError:
                            extra_data[str(key)] = str(value)
                diag_ent["attributes"].update(extra_data)

    if clusters_url:
        with open_url(clusters_url) as response:
            mimetype = get_mimetype(response)
            if mimetype is None:
                raise ValueError("Could not determine mimetype.")
            label_column = "label"
            for ent in ensure_dict(load_entities(response, mimetype=mimetype)):
                entity_id = ent.get("ID")
                if entity_id is None:
                    continue
                diag_ent = entities.get(str(entity_id), None)
                if diag_ent is None:
                    continue
                href = ent.get("href")
                if href:
                    diag_ent["href"] = href
                if label_column not in ent:
                    entity_columns = [col for col in ent.keys() if col not in {"ID", "href"}]
                    if len(entity_columns) != 1:
                        raise ValueError(
                            f"Unable to determine label column from {entity_columns}!"
                        )
                    label_column = entity_columns[0]
                diag_ent["label"] = str(ent[label_column])

    ent_list = list(entities.values())
    attributes: Set[str] = {
        attr
        for ent in ent_list
        for attr in ent.get("attributes", {}).keys()
        if attr is not None
    }
    reserved_column_names = {"ID", "name", "URL", "x", "y", "z", "Cluster ID", "size"}
    attr_to_column_name: Dict[str, str] = {}
    used_column_names: Set[str] = set(reserved_column_names)
    for attr in sorted(attributes):
        column_name = attr
        if column_name in used_column_names:
            index = 1
            while f"Entity {attr} ({index})" in used_column_names:
                index += 1
            column_name = f"Entity {attr} ({index})"
        attr_to_column_name[attr] = column_name
        used_column_names.add(column_name)

    df_data: Dict[str, Any] = {
        "ID": [e["ID"] for e in ent_list],
        "name": [e["name"] for e in ent_list],
        "URL": [e["href"] for e in ent_list],
        "x": [e["x"] for e in ent_list],
        "y": [e["y"] for e in ent_list],
        "z": [e["z"] if is_3d else 0 for e in ent_list],
        "Cluster ID": [e.get("label", "0") for e in ent_list],
        "size": [10 for _ in ent_list],
    }
    for attr, column_name in attr_to_column_name.items():
        df_data[column_name] = [
            e.get("attributes", {}).get(attr, None) for e in ent_list
        ]
    df = pd.DataFrame(df_data)

    hover_data = {"size": False, "URL": True, "z": is_3d}
    for column_name in attr_to_column_name.values():
        hover_data[column_name] = True

    if is_3d:
        fig = px.scatter_3d(
            df,
            x="x",
            y="y",
            z="z",
            size="size",
            hover_name="name",
            color="Cluster ID",
            symbol="Cluster ID",
            hover_data=hover_data,
            custom_data=["URL", "ID"],
        )
    else:
        fig = px.scatter(
            df,
            x="x",
            y="y",
            size="size",
            hover_name="name",
            color="Cluster ID",
            symbol="Cluster ID",
            hover_data=hover_data,
            custom_data=["URL", "ID"],
        )

    fig.update_layout(clickmode="event+select")

    post_script = """
const plotElement = document.getElementById('{plot_id}');
if (plotElement) {
  const linkContainerId = `${plotElement.id}-entity-link`;
  let linkContainer = document.getElementById(linkContainerId);
  if (!linkContainer) {
    linkContainer = document.createElement('div');
    linkContainer.id = linkContainerId;
    linkContainer.style.marginTop = '0.75rem';
    linkContainer.style.fontSize = '0.9rem';
    linkContainer.textContent = 'Selected entity link: n/a';
    plotElement.insertAdjacentElement('afterend', linkContainer);
  }

  const updateLink = (point) => {
    const customData = point && point.customdata ? point.customdata : [];
    const href = typeof customData[0] === 'string' ? customData[0].trim() : '';
    const entityId = customData[1] || point.hovertext || 'entity';

    linkContainer.replaceChildren();
    if (!href) {
      linkContainer.textContent = 'Selected entity link: n/a';
      return;
    }

    const label = document.createElement('span');
    label.textContent = `Selected entity (${entityId}): `;
    const link = document.createElement('a');
    link.href = href;
    link.target = '_blank';
    link.rel = 'noopener noreferrer';
    link.textContent = href;
    linkContainer.append(label, link);
  };

  plotElement.on('plotly_hover', (eventData) => {
    const point = eventData && eventData.points && eventData.points[0];
    if (point) {
      updateLink(point);
    }
  });
  plotElement.on('plotly_click', (eventData) => {
    const point = eventData && eventData.points && eventData.points[0];
    if (point) {
      updateLink(point);
    }
  });
}
"""

    return fig.to_html(full_html=full_html, post_script=post_script), Path(name).stem


@CELERY.task(
    name=f"{ClusterScatterVisualization.instance.identifier}.generate_plot", bind=True
)
def generate_plot(
    self,
    entity_url: str,
    clusters_url: Optional[str],
    entity_data_url: Optional[str],
    hash_: str,
) -> str:

    TASK_LOGGER.info(
        "Generating plot for entities %s, clusters %s and entity data %s...",
        entity_url,
        clusters_url,
        entity_data_url,
    )

    try:
        diagram, _ = _get_plot(
            entity_url=entity_url,
            clusters_url=clusters_url,
            entity_data_url=entity_data_url,
            full_html=False,
        )
    except HTTPError:
        DataBlob.set_value(ClusterScatterVisualization.instance.identifier, hash_, b"")
        PluginState.delete_value(
            ClusterScatterVisualization.instance.identifier, hash_, commit=True
        )
        return "Invalid Entity URL!"

    # Html needs to be saved as bytes, so it can be stored in a DataBlob
    html_bytes = diagram.encode(encoding="utf-8")

    DataBlob.set_value(ClusterScatterVisualization.instance.identifier, hash_, html_bytes)
    PluginState.delete_value(
        ClusterScatterVisualization.instance.identifier, hash_, commit=True
    )

    return "Created plot!"


@CELERY.task(name=f"{ClusterScatterVisualization.instance.identifier}.process", bind=True)
def process(
    self,
    db_id: str,
    entity_url: str,
    clusters_url: Optional[str],
    entity_data_url: Optional[str],
) -> str:
    diagram, name = _get_plot(
        entity_url=entity_url,
        clusters_url=clusters_url,
        entity_data_url=entity_data_url,
        full_html=True,
    )

    # Html needs to be saved as bytes, so it can be stored in a DataBlob
    html_bytes = diagram.encode(encoding="utf-8")

    with SpooledTemporaryFile() as output:
        output.write(html_bytes)
        output.seek(0)
        STORE.persist_task_result(
            db_id, output, f"plot_{name}.html", "image/html", "text/html"
        )
    return "Created plot!"

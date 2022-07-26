// Copyright 2021 QHAna plugin runner contributors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


/**
 * Register the main message event listener on the current window.
 */
function registerMessageListener() {
    // main event listener, delegates events to dedicated listeners
    window.addEventListener("message", (event) => {
        var data = event.data;
        if (typeof data === "string") {
            if (data === "ui-prevent-submit") {
                window._qhana_microfrontend_state.preventSubmit = true;
                sendSubmitStatus();
            }
            if (data === "ui-allow-submit") {
                window._qhana_microfrontend_state.preventSubmit = false;
                sendSubmitStatus();
            }
            if (data === "ui-submit-status") {
                sendSubmitStatus();
            }
        } else {
            if (data != null && data.type === "load-css") {
                onLoadCssMessage(data, window._qhana_microfrontend_state);
            }
            if (data != null && data.type === "data-url-response") {
                onDataUrlResponseMessage(data, window._qhana_microfrontend_state);
            }
            if (data != null && data.type === "plugin-url-response") {
                onPluginUrlResponseMessage(data, window._qhana_microfrontend_state);
            }
        }
    });
}

/**
 * Handle css load messages that request the micro frontend to load additional css files.
 *
 * @param {{type: 'load-css', urls: string[]}} data 
 * @param {{lastHeight: number, heightUnchangedCount: number}} state 
 */
function onLoadCssMessage(data, state) {
    var head = document.querySelector("head");
    data.urls.forEach((url) => {
        var styleLink = document.createElement("link");
        styleLink.href = url;
        styleLink.rel = "stylesheet";
        head.appendChild(styleLink);
    });
    state.heightUnchangedCount = 0;
    document.body.style.background = "transparent";
    monitorHeightChanges(state);
}

/**
 * Handle a data-url-response message containing a data URL for a specific form input.
 *
 * @param {{type: 'data-url-response', inputKey: string, href: string, dataType?: string, contentType?: string, filename?: string, version?: number}} data 
 * @param {{lastHeight: number, heightUnchangedCount: number}} state 
 */
function onDataUrlResponseMessage(data) {
    var input = document.querySelector(`input#${data.inputKey}`);
    if (input != null) {
        input.value = data.href;
        input.dispatchEvent(new InputEvent("input", { data: data.href, cancelable: false }));
        input.dispatchEvent(new InputEvent("change", { data: data.href, cancelable: false }));
        var filenameSpan = document.querySelector(`.selected-file-name[data-input-id=${data.inputKey}]`);
        if (filenameSpan != null) {
            filenameSpan.textContent = `${data.filename || "unknown"} (v${data.version || "?"}) ${data.dataType || "*"} – ${data.contentType || "*"}`;
            if (filenameSpan.parentNode.hasAttribute("hidden")) {
                filenameSpan.parentNode.removeAttribute("hidden");
                filenameSpan.parentNode.setAttribute("aria-live", "polite");
                monitorHeightChanges(window._qhana_microfrontend_state);
            }
        }
    }
}

/**
 * Handle a plugin-url-response message containing a plugin URL for a specific form input.
 *
 * @param {{type: 'plugin-url-response', inputKey: string, pluginUrl: string, pluginName?: string, pluginVersion?: string}} data 
 * @param {{lastHeight: number, heightUnchangedCount: number}} state 
 */
function onPluginUrlResponseMessage(data) {
    var input = document.querySelector(`input#${data.inputKey}`);
    if (input != null) {
        input.value = data.pluginUrl;
        input.dispatchEvent(new InputEvent("input", { data: data.pluginUrl, cancelable: false }));
        input.dispatchEvent(new InputEvent("change", { data: data.pluginUrl, cancelable: false }));
        var pluginNameSpan = document.querySelector(`.selected-plugin-name[data-input-id=${data.inputKey}]`);
        if (pluginNameSpan != null) {
            pluginNameSpan.textContent = `${data.pluginName || "unknown"} (v${data.pluginVersion || "?"})`;
            if (pluginNameSpan.parentNode.hasAttribute("hidden")) {
                pluginNameSpan.parentNode.removeAttribute("hidden");
                pluginNameSpan.parentNode.setAttribute("aria-live", "polite");
                monitorHeightChanges(window._qhana_microfrontend_state);
            }
        }
    }
}

/**
 * Send the current submit status to the plugin host.
 */
function sendSubmitStatus() {
    sendMessage({
        type: "ui-submit-status",
        isAllowedToSubmit: !window._qhana_microfrontend_state.preventSubmit,
    });
}

/**
 * Send a message to the parent window.
 *
 * @param {string|object} message the data attribute of the created message event
 */
function sendMessage(message) {
    var targetWindow = window.opener || window.parent;
    if (targetWindow) {
        targetWindow.postMessage(message, "*");
    } else {
        console.warn("Failed to message parent window. Is this page loaded outside of an iframe?");
    }
}

/**
 * Monitor height changes for a certain time and inform the parent window if the height has changed.
 *
 * @param {{lastHeight: number, heightUnchangedCount: number}} state 
 */
function monitorHeightChanges(state) {
    var newHeight = notifyParentWindowOnHeightChange(state.height);
    if (state.height === newHeight) {
        if (state.heightUnchangedCount > 60) { // allow for 60*50ms for height to settle
            return;
        }
        state.heightUnchangedCount = (state.heightUnchangedCount || 0) + 1;
    } else {
        state.heightUnchangedCount = 0;
        state.height = newHeight;
    }
    window.setTimeout(() => monitorHeightChanges(state), 50);
}

/**
 * Measure the current height and inform the parent window if the height has changed compared to `lastHeight`.
 *
 * @param {number} lastHeight the last measured height returned by this method (default: 0)
 * @returns the current measured height
 */
function notifyParentWindowOnHeightChange(lastHeight = 0) {
    var height = Math.max(document.body.scrollHeight, document.documentElement.scrollHeight);
    if (height !== lastHeight) {
        sendMessage({ type: "ui-resize", height: height });
    }
    return height;
}

/**
 * Notify the parent window that a micro frontend was successfully loaded and is available to receive messages.
 * 
 * Must be called **after** the message listener was attached to the window!
 */
function notifyParentWindowOnLoad() {
    sendMessage("ui-loaded");
    notifyParentWindowOnHeightChange();
}

/**
 * Instrument embedded html forms to listen for submit events.
 */
function instrumentForm() {
    const forms = document.querySelectorAll('form.qhana-form');
    forms.forEach(form => {
        const privateInputs = new Set(); // inputs that should be censored (e.g. password inputs)
        const dataInputs = new Set(); // inputs that contain urls pointing to data input
        form.querySelectorAll('input[data-private],input[type=password]').forEach(inputElement => {
            const name = inputElement.getAttribute('name');
            if (name == null) {
                console.warn('Input has no specified name but is marked as private!', inputElement);
            } else {
                privateInputs.add(name);
            }
        });
        form.querySelectorAll('input[data-input-type=data]').forEach(inputElement => {
            const name = inputElement.getAttribute('name');
            if (name == null) {
                console.warn('Input has no specified name but is marked as data input!', inputElement);
            } else {
                dataInputs.add(name);
            }
            var dataInputId = inputElement.getAttribute("id");
            var dataInputValue = inputElement.value;
            if (dataInputId && dataInputValue) {
                sendMessage({
                    type: "request-data-url-info",
                    inputKey: dataInputId,
                    dataUrl: dataInputValue,
                });
            }
        });
        form.querySelectorAll('input[data-input-type=plugin]').forEach(inputElement => {
            var pluginInputId = inputElement.getAttribute("id");
            var pluginInputValue = inputElement.value;
            if (pluginInputId && pluginInputValue) {
                sendMessage({
                    type: "request-plugin-url-info",
                    inputKey: pluginInputId,
                    pluginUrl: pluginInputValue,
                });
            }
        });
        form.querySelectorAll('button.qhana-choose-file-button').forEach(chooseButton => {
            var inputId = chooseButton.getAttribute("data-input-id");
            var dataType = chooseButton.getAttribute("data-input-type") ?? "*";
            var contentTypes = chooseButton.getAttribute("data-content-type");
            if (contentTypes == null) {
                contentTypes = ["*"];
            } else {
                contentTypes = contentTypes.split();
            }
            chooseButton.addEventListener("click", (event) => {
                event.preventDefault();
                sendMessage({
                    type: "request-data-url",
                    inputKey: inputId,
                    acceptedInputType: dataType,
                    acceptedContentTypes: contentTypes,
                });
            });
            chooseButton.removeAttribute("hidden");
        });
        form.querySelectorAll('button.qhana-choose-plugin-button').forEach(chooseButton => {
            var inputId = chooseButton.getAttribute("data-input-id");
            var pluginName = chooseButton.getAttribute("data-plugin-name") ?? null;
            var pluginVersion = chooseButton.getAttribute("data-plugin-version") ?? null;
            var pluginTags = chooseButton.getAttribute("data-plugin-tags") ?? "";
            if (pluginTags == null) {
                pluginTags = [];
            } else {
                pluginTags = pluginTags.split();
            }
            var requestMessage = {
                type: "request-plugin-url",
                inputKey: inputId,
                pluginTags: pluginTags,
            };
            if (pluginName) {
                requestMessage.pluginName = pluginName;
            }
            if (pluginVersion) {
                requestMessage.pluginVersion = pluginVersion;
            }
            chooseButton.addEventListener("click", (event) => {
                event.preventDefault();
                sendMessage(requestMessage);
            });
            chooseButton.removeAttribute("hidden");
        });
        form.addEventListener("submit", (event) => {
            onFormSubmit(event, dataInputs, privateInputs)
        });
    });
}

function onFormSubmit(event, dataInputs, privateInputs) {
    const form = event.target;
    const submitter = event.submitter;
    let submitTarget = submitter.getAttribute("data-target") || form.getAttribute("data-target");
    if (submitTarget === "microfrontend") {
        sendMessage("ui-loading");
        return; // will just be replaced with another micro frontend, nothing to do
    }
    if (submitTarget === "api") {
        event.preventDefault(); // cancel submit; need to do this manually...
        const formData = new FormData(form);
        let formMethod = form.method;
        let formAction = new URL(form.action);
        if (submitter != null) {
            formAction = new URL(submitter.formAction);
            formMethod = submitter.formMethod || formMethod;
        }
        formMethod = formMethod || "post";
        const submitUrl = formAction.toString();
        sendMessage("ui-loading");

        const inputDataUrls = new Set();
        const processedFormData = new FormData();
        formData.forEach((entry, key) => {
            if (privateInputs.has(key)) {
                // censor private values
                processedFormData.append(key, '***');
                return;
            }
            // add all other values unchanged
            processedFormData.append(key, entry);
            // add data inputs to extra list
            if (dataInputs.has(key) && (typeof entry === 'string')) {
                inputDataUrls.add(entry);
            }
        });

        if (window._qhana_microfrontend_state.preventSubmit) {
            sendMessage({
                type: "form-submit",
                formData: (new URLSearchParams(processedFormData)).toString(),
                formDataType: "application/x-www-form-urlencoded",
                dataInputs: new Array(...inputDataUrls),
                submitUrl: submitUrl,
            });
            return;
        }

        submitFormData(formData, formAction, formMethod)
            .then(
                (response) => {
                    if (response.status === 200) {
                        sendMessage({
                            type: "form-submit",
                            formData: (new URLSearchParams(processedFormData)).toString(),
                            formDataType: "application/x-www-form-urlencoded",
                            dataInputs: new Array(...inputDataUrls),
                            submitUrl: submitUrl,
                            resultUrl: response.url,
                        });
                    } else {
                        response.json().then((jsonBody) => {
                            sendMessage({
                                type: "form-error",
                                status: response.status,
                                error: jsonBody,
                            }, (error) => {
                                sendMessage({
                                    type: "form-error",
                                    status: response.status,
                                    error: "unknown error",
                                });

                            });
                        });
                    }
                },
                (error) => {
                    // TODO notify in case of error
                    sendMessage({
                        type: "form-error",
                        status: 500,
                        error: "unknown error",
                    });
                }
            );
    }
}

function submitFormData(formData, formAction, formMethod) {
    if (formMethod.toLowerCase() === "get") {
        let data = new URLSearchParams(formData);

        data.forEach((value, key) => {
            formAction.searchParams.append(key, value);
        })
        return fetch(formAction.toString())
    }

    return fetch(formAction.toString(), {
        body: formData,
        method: formMethod,
    });
}


// Main script entry point /////////////////////////////////////////////////////


// only execute functions if loaded from a parent window (e.g. inside an iframe)
if (window.top !== window.self) {
    if (window._qhana_microfrontend_state == null) {
        // prevent double execution if script is already loaded in the current window
        window._qhana_microfrontend_state = {
            href: window.location.href,
            lastHeight: 0,
            heightUnchangedCount: 0,
            preventSubmit: false,
        }
        instrumentForm();
        registerMessageListener();
        notifyParentWindowOnLoad();
    }
}

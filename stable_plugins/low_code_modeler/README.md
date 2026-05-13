This folder contains a plugin for the [low-code-modeler](https://github.com/LEQO-Framework/low-code-modeler)

To regenerate/update the static files you can just run `bash update-lcm.sh <path to low-code-modeler repo>`.

The shell script [`update-lcm.sh`](update-lcm.sh) builds the
low-code-modeler using `pnpm run build` and then injects the code in
[`head.html`](head.html) and [`body.html`](body.html) in the head and
the body of the generated `index.html` respectively.

To configure this plugin you can configure the services in the plugin
registry using the identifiers listed in [config.py](config.py)
(e.g. `QunicornEndpoint`) or environment variables, which use the same
names but have `LCM_` as a prefix (e.g. `LCM_QunicornEndpoint`)

This should be fixed by `workarounds.js`, but you may need to clear the
`workbox-precache` inside your browser to actually see any changes,
especially if you upgrade from a version without `workarounds.js`.

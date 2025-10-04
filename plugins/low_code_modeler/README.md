This folder contains a plugin for the [low-code-modeler](https://github.com/LEQO-Framework/low-code-modeler)

To regenerate/update the static files you have to run the following commands,
please run them from inside this directory, unless otherwise specified.

1. `git rm -rf static`
2. `git reset -- static/full-height.js static/clear-cache.js static/reset-style.css`
3. `git checkout -- static/full-height.js static/clear-cache.js static/reset-style.css`
4. `pnpm run build --outDir <path to qhana-plugin-runner>/plugins/low_code_modeler/static` (from inside the low-code-modeler repo)
5. `sed -i '/<\/script>/a <script defer src="/static/microfrontend.js"></script><script defer src="full-height.js"></script><script defer src="clear-cache.js"></script>' static/index.html`
6. `sed -i '/link rel="stylesheet"/a <link rel="stylesheet" href="reset-style.css">' static/index.html`
7. `git add static`

This should be fixed by `clear-cache.js`, but you may need to clear
the `workbox-precache` inside your browser to actually see any
changes, especially if you upgrade from a version without
`clear-cache.js`.

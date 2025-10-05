This folder contains a plugin for the [low-code-modeler](https://github.com/LEQO-Framework/low-code-modeler)

To regenerate/update the static files you have to run the following commands,
please run them from inside this directory, unless otherwise specified.

1. `git rm -rf static`
2. `git reset -- static/workarounds.js`
3. `git checkout -- static/workarounds.js`
4. `pnpm run build --outDir <path to qhana-plugin-runner>/plugins/low_code_modeler/static` (from inside the low-code-modeler repo)
5. `sed -i '/<\/script>/a <script defer src="/static/microfrontend.js"></script><script defer src="workarounds.js"></script>' static/index.html`
6. `git add static`

This should be fixed by `workarounds.js`, but you may need to clear the
`workbox-precache` inside your browser to actually see any changes,
especially if you upgrade from a version without `workarounds.js`.

# Incomplete sketch of instructions for updating static files

first run
`poetry run python -m static_patternatlas --atlas-url="http://localhost:1977/patternatlas" --out <path to qhana-plugin-runner>/plugins/pattern_atlas/static`
(from inside the `pattern-atlas-static` repo while the `pattern-atlas-api` is running)

then run this to patch generated html files:

```bash
for file in $(find static -name '*.html'); do
	relative_plugin_root="$(grep -o '<a class="home" href="\([^"]*\)">Pattern Atlas</a>' "$file" | sed 's/.*href="\([^"]*\)".*/\1/g')"
	sed -i '/<head>/a <script defer src="/static/microfrontend.js"></script><script src="'"$relative_plugin_root"'workarounds.js"></script>' "$file"
done
```

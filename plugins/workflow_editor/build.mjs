import * as esbuild from 'esbuild'

await esbuild.build({
    entryPoints: ['editor.js'],
    bundle: true,
    jsx: "automatic",
    sourcemap: "both",
    loader: {
        ".js": "js",
        ".css": "css",
        ".png": "file",
        ".svg": "file"
    },
    outfile: 'assets/editor-bundle.js',
})

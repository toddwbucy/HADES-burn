# Vendored frontend libraries

These are checked-in, pinned third-party JavaScript bundles. They are served
locally by `hades-viewer` (never fetched from a CDN at runtime — HADES forbids
runtime CDN dependencies). Re-vendor by repeating the steps below.

| File | Package | Version | License | Global |
|------|---------|---------|---------|--------|
| `graphology.min.js` | graphology | 0.26.0 | MIT | `window.graphology` (Graph classes) |
| `sigma.min.js` | sigma | 3.0.3 | MIT | `window.Sigma` (WebGL renderer) |
| `fa2.bundle.min.js` | graphology-layout-forceatlas2 | 0.10.1 | MIT | `window.HadesFA2.{forceAtlas2, FA2Layout}` |

## Re-vendoring

`graphology.min.js` and `sigma.min.js` are the packages' own UMD dist builds:

```
curl -o graphology.min.js https://cdn.jsdelivr.net/npm/graphology@0.26.0/dist/graphology.umd.min.js
curl -o sigma.min.js       https://cdn.jsdelivr.net/npm/sigma@3.0.3/dist/sigma.min.js
```

`graphology-layout-forceatlas2` ships no browser bundle (CommonJS only), so its
sync layout + Web Worker supervisor are bundled once with esbuild into an IIFE:

```
npm install graphology-layout-forceatlas2@0.10.1 graphology-utils
# entry.js:
#   import forceAtlas2 from 'graphology-layout-forceatlas2';
#   import FA2Layout   from 'graphology-layout-forceatlas2/worker';
#   export { forceAtlas2, FA2Layout };
npx esbuild entry.js --bundle --format=iife --global-name=HadesFA2 --minify \
    --outfile=fa2.bundle.min.js
```

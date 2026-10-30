# Documentation

The [public docs](https://wenkesj.github.io/trains/) cover datasets, examples,
and the Python reference. Edit Markdown in `src/content/docs/` and navigation in
`src/Documentation.tsx` and `src/lib/documentation.ts`.

## Preview

```sh
task docs:install docs:test docs:compile docs:serve
```

The public build is written to `docs/dist/`. Python reference content is generated
from Google-style docstrings with mkdocstrings. Run `task docs:pages` to build for
the GitHub Pages path. Deployment remains disabled until the repository is public
and `PAGES_ENABLED=true` is set.

## Development API documentation

Training and local inference documentation is private development material in
`development/`. It is excluded from the public navigation, search, JavaScript,
and build artifacts.

```sh
task docs:dev:compile docs:dev:serve
```

This separate build is written to `var/docs-development/` and served on loopback
port 8384. Keep it off public hosting; it has no authentication. The public docs
container and Pages workflow serve only `docs/dist/`.

After changing service endpoints, run `task docs:openapi docs:openapi:check` to
update and verify `development/openapi.json`.

To refresh the public docs container:

```sh
docker compose -f docs/compose.yml up -d --build docs
```

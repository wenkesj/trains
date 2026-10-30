/** Scalar renders live service schemas, with anchors inside the app's docs route. */
import { useMemo } from "react";
import { ApiReferenceReact, type AnyApiReferenceConfiguration } from "@scalar/api-reference-react";
import "@scalar/api-reference-react/style.css";
import { apiOperations, headingId, operationSlug, type LoadedApi } from "../lib/documentation";

const SDK_INSTALLATION = [
  { lang: "Python", description: "Install the Trains library from a checkout:\n\n```bash\nuv add --editable /path/to/trains/trains\n```\n\n```python\nfrom trains.datasets import chat_dataset_from_records\n```\n\n[Library setup and contracts →](#/docs/guides/sdk)" },
];

export default function ApiDocumentation({ api, focus }: { api: LoadedApi; focus?: string }) {
  const focused = apiOperations(api).find(({ method, path }) => {
    const slug = operationSlug(api.source.id, method, path);
    return focus === slug || focus?.startsWith(slug + "/");
  });
  const document = focused && api.document
    ? { ...api.document, paths: { [focused.path]: { [focused.method]: focused.operation } } }
    : api.document;
  const configuration = useMemo<AnyApiReferenceConfiguration>(() => ({
    content: document ? { ...document, info: { ...document.info, "x-scalar-sdk-installation": SDK_INSTALLATION } } : undefined,
    theme: "none",
    layout: "modern",
    showSidebar: false,
    hideSearch: true,
    hideDarkModeToggle: true,
    forceDarkModeState: "light",
    hideClientButton: true,
    showDeveloperTools: "never",
    hideTestRequestButton: true,
    withDefaultFonts: false,
    agent: { disabled: true },
    defaultHttpClient: { targetKey: "shell", clientKey: "curl" },
    hiddenClients: ["c", "clojure", "csharp", "dart", "fsharp", "go", "java", "js", "julia", "kotlin", "node", "objc", "ocaml", "php", "powershell", "python", "r", "ruby", "rust", "swift", "http1.1", "httpie", "wget"],
    generateHeadingSlug: ({ slug }) => `docs/reference/${api.source.id}/heading/${slug ?? "overview"}`,
    generateTagSlug: ({ name }) => `docs/reference/${api.source.id}/tag/${headingId(name ?? "default")}`,
    generateModelSlug: ({ name }) => `docs/reference/${api.source.id}/model/${headingId(name ?? "model")}`,
    generateOperationSlug: ({ method, path }) => operationSlug(api.source.id, method, path),
    generateWebhookSlug: ({ name }) => `docs/reference/${api.source.id}/webhook/${headingId(name)}`,
    redirect: (path) => {
      const matches = [...path.matchAll(new RegExp(`docs/reference/${api.source.id}/(?:operation|tag|model|heading|webhook)/[^/#]+`, "g"))];
      const target = matches.at(-1)?.[0];
      return target ? `#/${target}` : `#/docs/reference/${api.source.id}`;
    },
  }), [api, focus]);
  return <div className="docs-reference"><ApiReferenceReact configuration={configuration} /></div>;
}

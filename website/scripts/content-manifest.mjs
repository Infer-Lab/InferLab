import { readdir } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { siteBase } from '../site.config.mjs';

export const websiteRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
export const repositoryRoot = path.resolve(websiteRoot, '..');
export { siteBase };

const fixedEntries = [
  {
    source: 'README.md',
    target: 'src/content/docs/docs/getting-started/installation.md',
    description: 'Install InferLab and run the first workspace-oriented commands.',
  },
  {
    source: 'plugins/inferlab/skills/inferlab/references/workspace-authoring.md',
    target: 'src/content/docs/docs/guides/workspace-authoring/index.md',
    description: 'Choose the focused authoring reference for an InferLab workspace task.',
  },
  {
    source: 'plugins/inferlab/skills/inferlab/references/workspace-definition.md',
    target: 'src/content/docs/docs/guides/workspace-authoring/workspace-definition.md',
    description: 'Define, place, upgrade, and validate an InferLab workspace.',
  },
  {
    source: 'plugins/inferlab/skills/inferlab/references/execution-authoring.md',
    target: 'src/content/docs/docs/guides/workspace-authoring/execution-authoring.md',
    description: 'Author profiling, runtime-image, and invocation-patch behavior.',
  },
  {
    source: 'plugins/inferlab/skills/inferlab/references/eval-authoring.md',
    target: 'src/content/docs/docs/guides/workspace-authoring/eval-authoring.md',
    description: 'Author Eval task, dataset, and inference-request behavior.',
  },
  {
    source: 'plugins/inferlab/skills/inferlab/references/bench-authoring.md',
    target: 'src/content/docs/docs/guides/workspace-authoring/bench-authoring.md',
    description: 'Author serving-Bench load, source, session, metric, and SLO behavior.',
  },
  {
    source: 'docs/tui.md',
    target: 'src/content/docs/docs/guides/tui.md',
    description: 'Use the view-only workspace console.',
  },
  {
    source: 'docs/backend-support.md',
    target: 'src/content/docs/docs/reference/backend-support.md',
    description: 'Qualified backend capabilities exposed by InferLab.',
  },
];

async function renderedEntries(kind) {
  const sourceDirectory = path.join(repositoryRoot, 'docs', kind);
  const names = (await readdir(sourceDirectory))
    .filter((name) => name.endsWith('.md'))
    .sort();

  return names.map((name) => ({
    source: `docs/${kind}/${name}`,
    target: `src/content/docs/docs/architecture/${kind}/${name}`,
    description:
      kind === 'rfc'
        ? 'Current InferLab specification clause set.'
        : 'Current InferLab architecture decision record.',
  }));
}

export async function contentManifest() {
  return [
    ...fixedEntries,
    ...(await renderedEntries('rfc')),
    ...(await renderedEntries('adr')),
  ];
}

export function routeForTarget(target) {
  let route = target.replace(/^src\/content\/docs\//, '').replace(/\.md$/, '');
  route = route.replace(/\/index$/, '');
  return `${siteBase}/${route.toLowerCase()}/`.replace(/\/+/g, '/');
}

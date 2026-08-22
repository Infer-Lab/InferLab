import assert from 'node:assert/strict';
import test from 'node:test';
import { contentManifest } from './content-manifest.mjs';

test('projects the canonical workspace-authoring reference set as focused pages', async () => {
  const manifest = await contentManifest();
  const authoringEntries = manifest.filter((entry) =>
    entry.source.startsWith('plugins/inferlab/skills/inferlab/references/workspace-') ||
    entry.source.endsWith('/execution-authoring.md') ||
    entry.source.endsWith('/eval-authoring.md') ||
    entry.source.endsWith('/bench-authoring.md'),
  );

  assert.equal(
    manifest.some((entry) => entry.source === 'docs/workspace-authoring.md'),
    false,
  );

  assert.deepEqual(
    authoringEntries.map(({ source, target }) => ({ source, target })),
    [
      {
        source: 'plugins/inferlab/skills/inferlab/references/workspace-authoring.md',
        target: 'src/content/docs/docs/guides/workspace-authoring/index.md',
      },
      {
        source: 'plugins/inferlab/skills/inferlab/references/workspace-definition.md',
        target: 'src/content/docs/docs/guides/workspace-authoring/workspace-definition.md',
      },
      {
        source: 'plugins/inferlab/skills/inferlab/references/execution-authoring.md',
        target: 'src/content/docs/docs/guides/workspace-authoring/execution-authoring.md',
      },
      {
        source: 'plugins/inferlab/skills/inferlab/references/eval-authoring.md',
        target: 'src/content/docs/docs/guides/workspace-authoring/eval-authoring.md',
      },
      {
        source: 'plugins/inferlab/skills/inferlab/references/bench-authoring.md',
        target: 'src/content/docs/docs/guides/workspace-authoring/bench-authoring.md',
      },
    ],
  );
});

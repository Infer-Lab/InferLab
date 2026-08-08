import assert from 'node:assert/strict';
import { readFile, readdir } from 'node:fs/promises';
import test from 'node:test';

const semanticPalette = [
  '--sl-color-white',
  '--sl-color-gray-1',
  '--sl-color-gray-2',
  '--sl-color-gray-3',
  '--sl-color-gray-4',
  '--sl-color-gray-5',
  '--sl-color-gray-6',
  '--sl-color-black',
];

test('leaves Starlight theme palettes authoritative while retaining InferLab branding', async () => {
  const styles = new URL('../src/styles/', import.meta.url);
  const files = (await readdir(styles)).filter((name) => name.endsWith('.css'));
  const customCss = (
    await Promise.all(files.map((name) => readFile(new URL(name, styles), 'utf8')))
  ).join('\n');
  const starlightCss = await readFile(new URL('starlight.css', styles), 'utf8');

  for (const property of semanticPalette) {
    assert.doesNotMatch(customCss, new RegExp(`${property}\\s*:`));
  }
  assert.match(starlightCss, /--sl-color-accent:\s*var\(--inferlab-blue\)/);
  assert.match(starlightCss, /--sl-font:\s*Inter/);
});

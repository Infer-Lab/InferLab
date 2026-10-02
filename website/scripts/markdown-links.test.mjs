import assert from 'node:assert/strict';
import test from 'node:test';
import { alternateBody, markdownLinks } from '../src/markdown-links.mjs';

const site = { siteBase: '/InferLab', siteOrigin: 'https://infer-lab.github.io' };

test('relative links resolve against the HTML route of index and non-index pages', () => {
  assert.equal(
    alternateBody('[m](./backend-support/)', 'docs/reference', site),
    '[m](/InferLab/docs/reference/backend-support.md)',
  );
  assert.equal(
    alternateBody('[t](../tui/)', 'docs/guides/workspace-authoring/bench-authoring', site),
    '[t](/InferLab/docs/guides/workspace-authoring/tui.md)',
  );
});

test('anchors, absolute docs routes, and query strings are preserved', () => {
  assert.equal(alternateBody('[a](#refresh)', 'docs/guides/tui', site), '[a](#refresh)');
  assert.equal(
    alternateBody('[s](/InferLab/docs/guides/tui/?q=x#refresh)', 'docs', site),
    '[s](/InferLab/docs/guides/tui.md?q=x#refresh)',
  );
  assert.equal(
    alternateBody('[d](https://infer-lab.github.io/InferLab/docs/)', 'docs/concepts', site),
    '[d](/InferLab/docs.md)',
  );
});

test('titled, reference-style, and HTML links are rewritten', () => {
  assert.equal(
    alternateBody('[t](../guides/tui/ "TUI")', 'docs/concepts', site),
    '[t](/InferLab/docs/guides/tui.md "TUI")',
  );
  assert.equal(
    alternateBody('See [tui][g].\n\n[g]: ../guides/tui/ "TUI"\n', 'docs/concepts', site),
    'See [tui][g].\n\n[g]: /InferLab/docs/guides/tui.md "TUI"\n',
  );
  assert.equal(
    alternateBody('<p><a href="https://infer-lab.github.io/InferLab/docs/">Docs</a></p>\n', 'docs/getting-started/installation', site),
    '<p><a href="/InferLab/docs.md">Docs</a></p>\n',
  );
});

test('assets keep their path, and external links and code stay untouched', () => {
  assert.equal(
    alternateBody('![flow](./flow.svg)', 'docs/concepts', site),
    '![flow](/InferLab/docs/concepts/flow.svg)',
  );
  const untouched = [
    '[gh](https://github.com/Infer-Lab/InferLab)',
    'Write `[label](./target/)` literally.',
    '```md\n[label](./target/)\n```\n',
  ].join('\n\n');
  assert.equal(alternateBody(untouched, 'docs/concepts', site), untouched);
});

test('generator comments are dropped from the alternate body', () => {
  const body = '<!-- GENERATED: do not edit. Source: RFC-0011 -->\n<!-- SIGNATURE: sha256:00 -->\n\n> **Version:** 0.4.0\n';
  assert.equal(alternateBody(body, 'docs/architecture/rfc/rfc-0011', site), '> **Version:** 0.4.0\n');
});

test('markdownLinks reports every link destination outside code', () => {
  const links = markdownLinks(
    '[a](./x/) [b](https://example.com/) <a href="/InferLab/docs/y/">y</a>\n\n```\n[c](./z/)\n```\n',
  ).map((link) => link.url);
  assert.deepEqual(links, ['./x/', 'https://example.com/', '/InferLab/docs/y/']);
});

test('a link whose text holds an image rewrites both destinations', () => {
  assert.equal(
    alternateBody('[![logo](./logo.svg)](../guides/tui/)', 'docs/concepts', site),
    '[![logo](/InferLab/docs/concepts/logo.svg)](/InferLab/docs/guides/tui.md)',
  );
});

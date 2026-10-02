import assert from 'node:assert/strict';
import test from 'node:test';
import { sidebarOrder } from '../src/sidebar.mjs';

test('pages read in sidebar order, with the specification corpus last', () => {
  assert.deepEqual(
    sidebarOrder([
      'docs/architecture/adr/adr-0001',
      'docs/architecture/rfc/rfc-0001',
      'docs/reference/backend-support',
      'docs/architecture/rfc',
      'docs/guides/tui',
      'docs/guides/workspace-authoring/bench-authoring',
      'docs/guides/workspace-authoring',
      'docs/getting-started/installation',
      'docs/concepts',
      'docs',
    ]),
    [
      'docs',
      'docs/getting-started/installation',
      'docs/concepts',
      'docs/guides/workspace-authoring',
      'docs/guides/workspace-authoring/bench-authoring',
      'docs/guides/tui',
      'docs/reference/backend-support',
      'docs/architecture/rfc',
      'docs/architecture/rfc/rfc-0001',
      'docs/architecture/adr/adr-0001',
    ],
  );
});

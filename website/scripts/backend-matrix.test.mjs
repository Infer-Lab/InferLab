import assert from 'node:assert/strict';
import test from 'node:test';
import { readBackendMatrix, parseBackendMatrix } from '../src/lib/backend-matrix.mjs';

const fixture = `
## Serving And Control

| Capability | Alpha | Beta |
| --- | --- | --- |
| Integration package | \`inferlab-integration-alpha\` | \`inferlab-integration-beta\` |
| Single-node \`single\` topology | Qualified for the baseline below | Supported: one replica |
| Gateway-backed \`single\` | Supported: \`edge\`, from the \`edge-dist\` distribution | — |
| Multi-node replica | Supported | — |
| Disaggregated prefill/decode | Qualified | — |
| KV-transfer backend | Qualified: Mooncake and NIXL | — |
| P/D Router backend | Supported: \`builtin\`, \`alpha-router\`, and \`builtin\` again | — |

## Next Section
`;

test('derives each framework column from the serving table', () => {
  const matrix = parseBackendMatrix(fixture);

  assert.deepEqual(
    matrix.frameworks.map(({ id, label }) => ({ id, label })),
    [
      { id: 'alpha', label: 'Alpha' },
      { id: 'beta', label: 'Beta' },
    ],
  );
  const [alpha, beta] = matrix.frameworks;
  assert.deepEqual(alpha.topologies, {
    direct: { status: 'Qualified', backends: [] },
    gateway: { status: 'Supported', backends: ['edge'] },
    'multi-node': { status: 'Supported', backends: [] },
    'prefill-decode': { status: 'Qualified', backends: ['builtin', 'alpha-router'] },
  });
  assert.deepEqual(alpha.kvTransfer, ['NIXL', 'Mooncake']);
  assert.equal(beta.topologies.gateway.status, null);
  assert.equal(beta.topologies['prefill-decode'].status, null);
});

test('fails loudly when a row the simulator reads is missing', () => {
  assert.throws(
    () => parseBackendMatrix(fixture.replace(/\| Multi-node replica .*\n/, '')),
    /Multi-node replica/,
  );
});

test('reads the authoritative matrix into the simulator model', async () => {
  const matrix = await readBackendMatrix();
  const byId = Object.fromEntries(matrix.frameworks.map((framework) => [framework.id, framework]));

  assert.deepEqual(Object.keys(byId), [
    'vllm',
    'sglang',
    'tensorrt-llm',
    'tokenspeed',
    'specialized-engine',
  ]);
  for (const framework of matrix.frameworks) {
    for (const [topology, entry] of Object.entries(framework.topologies)) {
      assert.ok(
        entry.status === null || ['Qualified', 'Supported', 'Limited'].includes(entry.status),
        `${framework.id} ${topology}: unexpected status ${entry.status}`,
      );
      if (entry.status !== null && (topology === 'gateway' || topology === 'prefill-decode')) {
        assert.ok(entry.backends.length > 0, `${framework.id} ${topology}: no backend`);
      }
    }
  }
  assert.deepEqual(byId.vllm.topologies['prefill-decode'].backends, [
    'builtin',
    'vllm-router',
    'dynamo',
  ]);
  assert.deepEqual(byId.vllm.topologies.gateway.backends, ['dynamo']);
  assert.deepEqual(byId['specialized-engine'].topologies.gateway.backends, ['smg']);
});

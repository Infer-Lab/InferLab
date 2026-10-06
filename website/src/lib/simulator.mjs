// The control-plane simulator on the product page. Pure functions render one
// selection — framework, topology, frontend backend — into the process
// topology, the workspace declaration that asks for it, and an illustrative
// lifecycle. The page renders the default selection on the server and the
// browser re-renders with the same functions, so both stay one implementation.
// Which selections exist comes only from the backend support matrix
// (./backend-matrix.mjs); this module adds presentation, not capability.

/**
 * @typedef {import('./backend-matrix.mjs').TopologyId} TopologyId
 * @typedef {import('./backend-matrix.mjs').Framework} Framework
 * @typedef {import('./backend-matrix.mjs').BackendMatrix} BackendMatrix
 * @typedef {{ framework: string, topology: TopologyId | undefined, backend: string | null, status: string | null, transport: string | null }} Selection
 */

/** @type {{ id: TopologyId, label: string, detail: string }[]} */
export const topologies = [
  { id: 'direct', label: 'Direct', detail: 'one Engine replica' },
  { id: 'gateway', label: 'Gateway', detail: 'routed single' },
  { id: 'prefill-decode', label: 'Prefill / decode', detail: 'disaggregated' },
  { id: 'multi-node', label: 'Multi-node', detail: 'one replica, two machines' },
];

/** Backend names whose frontend finds workers through a discovery process. */
const discoveringBackends = new Set(['dynamo']);

/** @param {Framework} framework */
export function availableTopologies(framework) {
  return topologies.filter((topology) => framework.topologies[topology.id].status !== null);
}

/**
 * Normalize a requested selection onto one the matrix offers.
 * @param {BackendMatrix} matrix
 * @param {{ framework?: string, topology?: string, backend?: string | null }} [requested]
 * @returns {Selection}
 */
export function resolveSelection(matrix, requested = {}) {
  const framework =
    matrix.frameworks.find((candidate) => candidate.id === requested.framework) ??
    matrix.frameworks[0];
  const offered = availableTopologies(framework);
  const topology =
    offered.find((candidate) => candidate.id === requested.topology)?.id ?? offered[0]?.id;
  const backends = topology ? framework.topologies[topology].backends : [];
  const backend = backends.includes(requested.backend) ? requested.backend : backends[0] ?? null;
  const entry = topology ? framework.topologies[topology] : undefined;
  // A published pairing table is the authority for one backend's status and
  // the transport that status was earned with.
  const pairing = backend !== null ? entry?.pairings?.[backend] : undefined;
  return {
    framework: framework.id,
    topology,
    backend,
    status: pairing?.status ?? entry?.status ?? null,
    transport:
      topology === 'prefill-decode'
        ? (pairing?.transport ?? framework.kvTransfer[0] ?? 'NIXL').toLowerCase()
        : null,
  };
}

const escape = (text) =>
  String(text).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');

function node({ id, x, y, w = 132, h = 58, kind, title, detail, tone = 'engine' }) {
  return `<g class="sim-node sim-node--${tone}" data-node="${id}" transform="translate(${x} ${y})">
    <rect class="sim-node__body" width="${w}" height="${h}" rx="10" />
    <rect class="sim-node__rail" width="3" height="${h - 20}" x="0" y="10" rx="1.5" />
    <text class="sim-node__kind" x="14" y="21">${escape(kind)}</text>
    <text class="sim-node__title" x="14" y="41">${escape(title)}</text>
    ${detail ? `<text class="sim-node__detail" x="${w - 12}" y="21" text-anchor="end">${escape(detail)}</text>` : ''}
  </g>`;
}

function edge({ id, d, kind = 'request', label, lx, ly }) {
  return `<g class="sim-edge sim-edge--${kind}">
    <path id="${id}" data-flow="${kind}" d="${d}" />
    ${label ? `<text class="sim-edge__label" x="${lx}" y="${ly}" text-anchor="middle">${escape(label)}</text>` : ''}
  </g>`;
}

function machine({ x, y, w, h, label }) {
  return `<g class="sim-machine"><rect x="${x}" y="${y}" width="${w}" height="${h}" rx="14" />
    <text x="${x + 14}" y="${y + 20}">${escape(label)}</text></g>`;
}

/** The process topology one selection resolves to, as inline SVG. */
export function renderTopology(matrix, selection) {
  const framework = matrix.frameworks.find((candidate) => candidate.id === selection.framework);
  const backend = selection.backend;
  const discovering = discoveringBackends.has(backend);
  const parts = [];
  const client = node({ id: 'client', x: 18, y: 141, w: 104, kind: 'CLIENT', title: 'workload', tone: 'client' });
  switch (selection.topology) {
    case 'direct':
      parts.push(
        edge({ id: 'e-client-serve', d: 'M122 170 C 230 170, 300 170, 392 170' }),
        client,
        node({ id: 'serve', x: 392, y: 141, w: 168, kind: 'SERVE · RANK 0', title: framework.label, detail: 'GPU×N' }),
      );
      break;
    case 'gateway':
      parts.push(
        edge({ id: 'e-client-gw', d: 'M122 170 C 170 170, 190 170, 238 170' }),
        edge({ id: 'e-gw-serve', d: 'M370 170 C 410 170, 420 170, 462 170' }),
        client,
        node({ id: 'gateway', x: 238, y: 141, kind: 'GATEWAY', title: backend, tone: 'frontend' }),
        node({ id: 'serve', x: 462, y: 141, w: 156, kind: 'SERVE · RANK 0', title: framework.label, detail: 'GPU×N' }),
      );
      break;
    case 'prefill-decode':
      parts.push(
        edge({ id: 'e-client-fe', d: 'M122 170 C 160 170, 180 170, 216 170' }),
        edge({ id: 'e-fe-prefill', d: 'M372 160 C 410 160, 410 86, 452 86' }),
        edge({ id: 'e-fe-decode', d: 'M372 180 C 410 180, 410 254, 452 254' }),
        edge({ id: 'e-kv', d: 'M535 116 C 535 160, 535 180, 535 224', kind: 'kv', label: 'KV transfer', lx: 580, ly: 174 }),
        client,
        node({ id: 'frontend', x: 216, y: 141, w: 156, kind: 'GATEWAY + P/D ROUTER', title: backend, tone: 'frontend' }),
        node({ id: 'prefill', x: 452, y: 57, w: 166, kind: 'PREFILL', title: framework.label, detail: 'GPU×N' }),
        node({ id: 'decode', x: 452, y: 225, w: 166, kind: 'DECODE', title: framework.label, detail: 'GPU×N' }),
      );
      break;
    case 'multi-node':
      parts.push(
        machine({ x: 300, y: 28, w: 330, h: 120, label: 'node-a' }),
        machine({ x: 300, y: 192, w: 330, h: 120, label: 'node-b' }),
        edge({ id: 'e-client-r0', d: 'M122 170 C 220 170, 230 96, 352 96' }),
        edge({ id: 'e-group', d: 'M450 125 C 450 170, 450 190, 450 239', kind: 'collective', label: 'tensor-parallel group', lx: 545, ly: 177 }),
        client,
        node({ id: 'rank0', x: 352, y: 67, w: 196, kind: 'RANK 0 · ENTRY', title: framework.label, detail: 'GPU×N' }),
        node({ id: 'rank1', x: 352, y: 239, w: 196, kind: 'RANK 1 · HEADLESS', title: framework.label, detail: 'GPU×N' }),
      );
      break;
    default:
      break;
  }
  if (discovering) {
    const target = selection.topology === 'prefill-decode' ? 'frontend' : 'gateway';
    const x = target === 'frontend' ? 236 : 248;
    parts.unshift(
      edge({ id: 'e-discovery', d: `M${x + 56} 289 C ${x + 56} 260, ${x + 56} 230, ${x + 56} 199`, kind: 'discovery', label: 'registry', lx: x + 98, ly: 250 }),
    );
    parts.push(node({ id: 'discovery', x, y: 289, w: 112, h: 46, kind: 'DISCOVERY', title: 'etcd', tone: 'discovery' }));
  }
  return `<svg class="sim-topology" viewBox="0 0 640 344" role="img" aria-label="${escape(
    describe(matrix, selection),
  )}">
    <defs>
      <linearGradient id="sim-request" x1="0" x2="1"><stop offset="0" stop-color="var(--sim-request-a)" /><stop offset="1" stop-color="var(--sim-request-b)" /></linearGradient>
      <filter id="sim-glow" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="3" result="b" /><feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge></filter>
    </defs>
    ${parts.join('\n')}
    <g class="sim-particles"></g>
  </svg>`;
}

/** A plain-language summary of the topology for assistive technology. */
export function describe(matrix, selection) {
  const framework = matrix.frameworks.find((candidate) => candidate.id === selection.framework);
  switch (selection.topology) {
    case 'direct':
      return `${framework.label} serving one Engine replica directly.`;
    case 'gateway':
      return `${framework.label} behind a ${selection.backend} Gateway.`;
    case 'prefill-decode':
      return `${framework.label} prefill and decode replicas behind a fused ${selection.backend} Gateway and P/D Router, with KV transfer from prefill to decode.`;
    case 'multi-node':
      return `${framework.label} serving one replica whose two ranks run on two machines.`;
    default:
      return framework.label;
  }
}

const key = (text) => `<span class="tok-key">${escape(text)}</span>`;
const str = (text) => `<span class="tok-str">"${escape(text)}"</span>`;
const num = (text) => `<span class="tok-num">${escape(text)}</span>`;
const table = (text) => `<span class="tok-table">[${escape(text)}]</span>`;
const comment = (text) => `<span class="tok-comment"># ${escape(text)}</span>`;

/** The workspace declaration, plus local placement where the shape needs it. */
export function renderDeclaration(matrix, selection) {
  const framework = matrix.frameworks.find((candidate) => candidate.id === selection.framework);
  const lines = [
    comment('.inferlab/workspace.d/demo.toml'),
    table(`stacks.${framework.id}`),
    `${key('integration')} = ${str(framework.id)}`,
    '',
    table('servers.demo'),
    `${key('stack')} = ${str(framework.id)}`,
    `${key('model')} = ${str('example')}`,
  ];
  switch (selection.topology) {
    case 'direct':
      lines.push(`${key('topology')} = ${str('single')}`);
      break;
    case 'gateway':
      lines.push(`${key('topology')} = ${str('single')}`, `${key('gateway_backend')} = ${str(selection.backend)}`);
      break;
    case 'prefill-decode': {
      const transport = selection.transport ?? (framework.kvTransfer[0] ?? 'NIXL').toLowerCase();
      lines.push(
        `${key('topology')} = ${str('prefill_decode')}`,
        `${key('gateway_backend')} = ${str(selection.backend)}`,
        `${key('pd_router_backend')} = ${str(selection.backend)}`,
        `${key('kv_transfer')} = ${str(transport)}`,
      );
      break;
    }
    case 'multi-node':
      lines.push(
        `${key('topology')} = ${str('single')}`,
        '',
        table('servers.demo.parallelism.outer'),
        `${key('tensor_parallel_size')} = ${num('2')}`,
        '',
        comment('.inferlab/local.toml — machine-private placement'),
        table('placements.two-node.roles.serve'),
        `${key('ranks')} = [`,
        `  { ${key('machine')} = ${str('node-a')}, ${key('devices')} = [${num('0')}] },`,
        `  { ${key('machine')} = ${str('node-b')}, ${key('devices')} = [${num('0')}] },`,
        ']',
      );
      break;
    default:
      break;
  }
  return lines.join('\n');
}

/** An illustrative lifecycle for the selection, phase by phase. */
export function lifecycle(selection) {
  const discovering = discoveringBackends.has(selection.backend);
  const steps = [
    { phase: 'resolve', text: 'plan_serve → render_serve · effective commands frozen' },
  ];
  if (discovering) steps.push({ phase: 'launch', text: 'discovery first · etcd ready' });
  switch (selection.topology) {
    case 'direct':
      steps.push({ phase: 'launch', text: 'serve rank 0 · readiness verified' });
      break;
    case 'gateway':
      steps.push(
        { phase: 'launch', text: 'serve rank 0 · readiness verified' },
        { phase: 'launch', text: `gateway ${selection.backend} · every target registered` },
      );
      break;
    case 'prefill-decode':
      steps.push(
        { phase: 'launch', text: 'prefill + decode · readiness verified' },
        { phase: 'launch', text: `frontend ${selection.backend} · every target registered` },
      );
      break;
    case 'multi-node':
      steps.push({ phase: 'launch', text: 'rank 0 node-a · rank 1 node-b · group joined' });
      break;
    default:
      break;
  }
  steps.push(
    { phase: 'measure', text: 'eval smoke → bench · raw + normalized metrics' },
    { phase: 'record', text: '.inferlab/records/<id>-recipe-demo-qualify-…/' },
    { phase: 'cleanup', text: 'process groups gone · devices freed' },
  );
  return steps;
}

// The product page's simulator offers only what the backend support matrix
// reports (RFC-0011:C-CONTENT-AUTHORITY): this module derives each framework's
// serving topologies, their status labels, and their frontend backend names
// from the matrix's "Serving And Control" table at build time. A row the
// simulator reads that goes missing fails the build instead of silently
// offering a stale capability.
import { readFile } from 'node:fs/promises';
import path from 'node:path';

// Resolved from the website root, the working directory of every website
// task: once bundled, this module no longer sits beside its source path.
const matrixPath = path.resolve(
  process.cwd(),
  '../plugins/inferlab/skills/inferlab/references/backend-support.md',
);

/**
 * @typedef {'direct' | 'gateway' | 'multi-node' | 'prefill-decode'} TopologyId
 * @typedef {{ transport: string, status: string }} Pairing
 * @typedef {{ status: string | null, backends: string[], pairings?: Record<string, Pairing> }} TopologyEntry
 * @typedef {{ id: string, label: string, topologies: Record<TopologyId, TopologyEntry>, kvTransfer: string[] }} Framework
 * @typedef {{ frameworks: Framework[] }} BackendMatrix
 */

/** The status words the matrix defines, in the order its legend lists them. */
const statuses = ['Qualified', 'Supported', 'Limited', 'Unsupported', 'Inconclusive', 'Unqualified'];

/** Simulator topology → the matrix row that reports it. */
const topologyRows = {
  direct: 'Single-node `single` topology',
  gateway: 'Gateway-backed `single`',
  'multi-node': 'Multi-node replica',
  'prefill-decode': 'Disaggregated prefill/decode',
};

const kvTransports = ['NIXL', 'Mooncake'];

function cells(line) {
  return line
    .trim()
    .replace(/^\|/, '')
    .replace(/\|$/, '')
    .split('|')
    .map((cell) => cell.trim());
}

function status(cell) {
  const word = statuses.find((candidate) => cell.startsWith(candidate));
  return word ?? null;
}

function backticked(cell) {
  return [...new Set([...cell.matchAll(/`([^`]+)`/g)].map((match) => match[1]))];
}

/**
 * The per-backend P/D pairing table a framework may publish ("### <label>
 * P/D Pairings"): each backend's best-status transport.
 * @param {string} markdown
 * @param {string} label
 * @param {string[]} backends
 * @returns {Record<string, Pairing> | null}
 */
function pairingTable(markdown, label, backends) {
  const section = markdown.split(`### ${label} P/D Pairings`)[1]?.split(/^#{2,3} /m)[0];
  if (section === undefined) {
    return null;
  }
  const lines = section.split('\n').filter((line) => line.startsWith('|'));
  const transports = cells(lines[0] ?? '').slice(1);
  /** @type {Record<string, Pairing>} */
  const pairings = {};
  for (const line of lines.slice(2)) {
    const [pair, ...values] = cells(line);
    const backend = pair.startsWith('Built-in')
      ? 'builtin'
      : pair.toLowerCase().replaceAll(' ', '-');
    if (!backends.includes(backend)) {
      throw new Error(`backend support matrix: ${label} pairing "${pair}" names no P/D backend`);
    }
    const best = transports
      .map((transport, index) => ({ transport, status: status(values[index] ?? '') }))
      .filter((candidate) => candidate.status !== null)
      .sort((a, b) => statuses.indexOf(a.status) - statuses.indexOf(b.status))[0];
    if (best !== undefined) {
      pairings[backend] = /** @type {Pairing} */ (best);
    }
  }
  return pairings;
}

/**
 * @param {string} markdown the backend support matrix
 * @returns {BackendMatrix}
 */
export function parseBackendMatrix(markdown) {
  const section = markdown.split(/^## Serving And Control$/m)[1]?.split(/^## /m)[0];
  if (section === undefined) {
    throw new Error('backend support matrix: missing the "Serving And Control" section');
  }
  const lines = section.split('\n').filter((line) => line.startsWith('|'));
  const header = cells(lines[0] ?? '');
  if (header[0] !== 'Capability') {
    throw new Error('backend support matrix: the serving table must start with a Capability column');
  }
  const rows = new Map(lines.slice(2).map((line) => {
    const [capability, ...values] = cells(line);
    return [capability, values];
  }));
  const row = (capability) => {
    const values = rows.get(capability);
    if (values === undefined || values.length !== header.length - 1) {
      throw new Error(`backend support matrix: missing or malformed row "${capability}"`);
    }
    return values;
  };

  const packages = row('Integration package');
  const publicComponentRow = row('`single` public component');
  const gatewayRow = row(topologyRows.gateway);
  const pdRouterRow = row('P/D Router backend');
  const kvRow = row('KV-transfer backend');
  const topologyValues = Object.fromEntries(
    Object.entries(topologyRows).map(([topology, capability]) => [topology, row(capability)]),
  );

  const frameworks = header.slice(1).map((label, index) => {
    const id = packages[index].match(/`inferlab-integration-([^`]+)`/)?.[1];
    if (id === undefined) {
      throw new Error(`backend support matrix: no integration package for ${label}`);
    }
    const topologies = /** @type {Record<TopologyId, TopologyEntry>} */ ({});
    for (const topology of /** @type {TopologyId[]} */ (Object.keys(topologyRows))) {
      const cell = topologyValues[topology][index];
      // A direct topology exists only where the public component is the
      // Engine itself; an Engine behind a required Gateway is not direct.
      const entryStatus =
        topology === 'direct' && !publicComponentRow[index].includes('direct-Engine')
          ? null
          : status(cell);
      let backends = [];
      if (entryStatus !== null && topology === 'gateway') {
        // The Gateway backend leads its cell; later names are distributions
        // and evidence detail.
        backends = backticked(gatewayRow[index]).slice(0, 1);
      } else if (entryStatus !== null && topology === 'prefill-decode') {
        backends = backticked(pdRouterRow[index]);
      }
      topologies[topology] = { status: entryStatus, backends };
      const pairings = topology === 'prefill-decode' ? pairingTable(markdown, label, backends) : null;
      if (entryStatus !== null && pairings !== null) {
        topologies[topology].pairings = pairings;
      }
    }
    return {
      id,
      label,
      topologies,
      kvTransfer: kvTransports.filter((transport) => kvRow[index].includes(transport)),
    };
  });
  return { frameworks };
}

/** @returns {Promise<BackendMatrix>} */
export async function readBackendMatrix() {
  return parseBackendMatrix(await readFile(matrixPath, 'utf8'));
}

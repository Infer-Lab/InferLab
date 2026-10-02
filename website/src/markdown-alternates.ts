// Plain-text Markdown alternates of every documentation page, plus the
// llms.txt index and llms-full.txt concatenation, generated from the same
// content collection the HTML pages render (RFC-0011:C-CONTENT-AUTHORITY,
// ADR-0054). Nothing here is independently editable.
import { getCollection, type CollectionEntry } from 'astro:content';
import { siteBase, siteOrigin } from '../site.config.mjs';
import { alternateBody } from './markdown-links.mjs';
import { sidebarOrder } from './sidebar.mjs';

type DocsEntry = CollectionEntry<'docs'>;

/** Every documentation page, in the order the sidebar lists them. */
export async function documentationPages(): Promise<DocsEntry[]> {
  const entries = new Map((await getCollection('docs')).map((entry) => [entry.id, entry]));
  return sidebarOrder([...entries.keys()]).flatMap((id) => entries.get(id) ?? []);
}

/** A specification record: an individual RFC or ADR page, not its index. */
export function isSpecificationRecord(entry: DocsEntry): boolean {
  return /^docs\/architecture\/(?:rfc|adr)\/./.test(entry.id);
}

/** The alternate's absolute URL: the page route with a `.md` suffix. */
export function alternateUrl(entry: DocsEntry): string {
  return new URL(`${siteBase}/${entry.id}.md`, siteOrigin).href;
}

/** The page title as its level-one heading, followed by the rendered body. */
export function markdownAlternate(entry: DocsEntry): string {
  const body = alternateBody((entry.body ?? '').trim(), entry.id, { siteBase, siteOrigin });
  return `# ${entry.data.title}\n\n${body.trim()}\n`;
}

// Link handling for the plain-text Markdown alternates (RFC-0011:C-CONTENT-AUTHORITY,
// ADR-0054), shared by the alternate generator and the build verifier so both
// see the same links. Links come from the Markdown syntax tree, so code spans
// and fenced code are never touched, and each destination is spliced in place
// so the rest of the page keeps its original text.
import { fromMarkdown } from 'mdast-util-from-markdown';
import { visit } from 'unist-util-visit';

const GENERATOR_COMMENTS = /^(?:<!-- (?:GENERATED|SIGNATURE): [^\n]*-->\r?\n)+\s*/;
const HTML_REFERENCE = /\b(?:href|src)="([^"]*)"/g;

/** Every link destination in a Markdown body, with its source offsets. */
export function markdownLinks(body) {
  const links = [];
  visit(fromMarkdown(body), (node) => {
    const start = node.position?.start.offset;
    const end = node.position?.end.offset;
    if (start === undefined || end === undefined) return;
    if (node.type === 'html') {
      for (const match of node.value.matchAll(HTML_REFERENCE)) {
        const offset = start + match.index + match[0].indexOf('"') + 1;
        links.push({ url: match[1], start: offset, end: offset + match[1].length });
      }
      return;
    }
    let destination;
    if (node.type === 'link' || node.type === 'image') {
      // An autolink `<url>` carries its destination as its text.
      if (body[start] === '<') return;
      // A link's text may itself hold a link or image, whose `](` comes
      // first; the destination follows the last child.
      const textEnd = node.children?.at(-1)?.position?.end.offset ?? start;
      const open = body.indexOf('](', textEnd);
      if (open === -1 || open >= end) return;
      destination = open + 2;
    } else if (node.type === 'definition') {
      const open = body.indexOf(']:', start);
      if (open === -1 || open >= end) return;
      destination = open + 2;
    } else {
      return;
    }
    while (/\s/.test(body[destination])) destination += 1;
    let stop;
    if (body[destination] === '<') {
      destination += 1;
      stop = body.indexOf('>', destination);
    } else {
      stop = destination;
      while (stop < end && !/[\s)]/.test(body[stop])) stop += 1;
    }
    if (stop <= destination) return;
    links.push({ url: body.slice(destination, stop), start: destination, end: stop });
  });
  return links.sort((left, right) => left.start - right.start);
}

/**
 * The plain-text form of one site link: resolved against the page's HTML
 * route, because the alternate lives one level higher than that route and
 * llms-full.txt concatenates it with every other page, and made root-relative.
 * A documentation page route points at its Markdown alternate, so an agent
 * following a link stays in plain text. Other links are returned unchanged.
 */
export function alternateLink(url, entryId, { siteBase, siteOrigin }) {
  if (url.startsWith('#')) return url;
  let resolved;
  try {
    resolved = new URL(url, new URL(`${siteBase}/${entryId}/`, siteOrigin));
  } catch {
    return url;
  }
  if (resolved.origin !== siteOrigin) return url;
  let pathname = resolved.pathname;
  const docsRoot = `${siteBase}/docs`;
  const lastSegment = pathname.replace(/\/$/, '').split('/').at(-1) ?? '';
  if ((pathname === docsRoot || pathname.startsWith(`${docsRoot}/`)) && !lastSegment.includes('.')) {
    pathname = `${pathname.replace(/\/$/, '')}.md`;
  }
  return `${pathname}${resolved.search}${resolved.hash}`;
}

/** A page body as its alternate carries it: generator comments dropped, site links rewritten. */
export function alternateBody(body, entryId, site) {
  let text = body.replace(GENERATOR_COMMENTS, '');
  for (const link of markdownLinks(text).reverse()) {
    text = text.slice(0, link.start) + alternateLink(link.url, entryId, site) + text.slice(link.end);
  }
  return text;
}

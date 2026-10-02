import type { APIRoute } from 'astro';
import { alternateUrl, documentationPages, isSpecificationRecord } from '../markdown-alternates';
import { siteBase, siteDescription, siteOrigin } from '../../site.config.mjs';

// The llms.txt convention: a title, a one-line summary, and sections of
// links to plain-text pages. The specification records go under `Optional`,
// the section an agent with a short context may skip.
export const GET: APIRoute = async () => {
  const pages = await documentationPages();
  const link = (entry: (typeof pages)[number]) => {
    const description = entry.data.description ? `: ${entry.data.description}` : '';
    return `- [${entry.data.title}](${alternateUrl(entry)})${description}`;
  };
  const full = new URL(`${siteBase}/llms-full.txt`, siteOrigin).href;
  const text = [
    '# InferLab',
    '',
    `> ${siteDescription}`,
    '',
    `Every documentation page below is plain-text Markdown; ${full} concatenates them all in this order.`,
    '',
    '## Documentation',
    '',
    ...pages.filter((entry) => !isSpecificationRecord(entry)).map(link),
    '',
    '## Optional',
    '',
    ...pages.filter(isSpecificationRecord).map(link),
    '',
  ].join('\n');
  return new Response(text, { headers: { 'Content-Type': 'text/plain; charset=utf-8' } });
};

import type { APIRoute } from 'astro';
import { alternateUrl, documentationPages, markdownAlternate } from '../markdown-alternates';

export const GET: APIRoute = async () => {
  const pages = await documentationPages();
  const text = pages
    .map((entry) => `<!-- ${alternateUrl(entry)} -->\n\n${markdownAlternate(entry)}`)
    .join('\n---\n\n');
  return new Response(text, { headers: { 'Content-Type': 'text/plain; charset=utf-8' } });
};

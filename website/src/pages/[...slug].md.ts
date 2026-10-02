import type { APIRoute, GetStaticPaths } from 'astro';
import type { CollectionEntry } from 'astro:content';
import { documentationPages, markdownAlternate } from '../markdown-alternates';

export const getStaticPaths: GetStaticPaths = async () =>
  (await documentationPages()).map((entry) => ({
    params: { slug: entry.id },
    props: { entry },
  }));

export const GET: APIRoute = ({ props }) =>
  new Response(markdownAlternate(props.entry as CollectionEntry<'docs'>), {
    headers: { 'Content-Type': 'text/markdown; charset=utf-8' },
  });

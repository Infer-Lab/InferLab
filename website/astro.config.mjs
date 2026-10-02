import { defineConfig } from 'astro/config';
import sitemap from '@astrojs/sitemap';
import starlight from '@astrojs/starlight';
import {
  repositoryUrl,
  siteBase,
  siteDescription,
  siteBaseWithSlash,
  siteOrigin,
} from './site.config.mjs';
import { sidebar } from './src/sidebar.mjs';

export default defineConfig({
  site: siteOrigin,
  base: siteBase,
  trailingSlash: 'ignore',
  integrations: [
    sitemap({
      filter: (page) => page !== `${siteOrigin}${siteBase}`,
      customPages: [`${siteOrigin}${siteBaseWithSlash}`],
    }),
    starlight({
      title: 'InferLab',
      description: siteDescription,
      favicon: '/favicon.svg',
      customCss: ['./src/styles/brand.css', './src/styles/starlight.css'],
      components: {
        SiteTitle: './src/components/StarlightSiteTitle.astro',
      },
      social: [
        {
          icon: 'github',
          label: 'InferLab on GitHub',
          href: repositoryUrl,
        },
      ],
      sidebar,
    }),
  ],
});

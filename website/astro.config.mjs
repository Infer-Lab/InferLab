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
        Header: './src/components/docs/Header.astro',
        PageTitle: './src/components/docs/PageTitle.astro',
        SiteTitle: './src/components/StarlightSiteTitle.astro',
      },
      expressiveCode: {
        styleOverrides: {
          borderRadius: '0.8rem',
          borderColor: 'var(--sl-color-gray-5)',
          frames: { shadowColor: 'transparent' },
        },
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

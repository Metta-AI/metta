#!/usr/bin/env node

import fs from 'fs';
import yaml from 'js-yaml';

// Read the links config
const linksConfig = yaml.load(fs.readFileSync('links.yaml', 'utf8'));

// Generate nginx location blocks for redirects
const allRedirects = [];

// Process main links and their sub-links
linksConfig.links.forEach(link => {
  // Add main link redirects (now an array)
  if (link.short_urls && link.short_urls.length > 0) {
    const url = link.url.replace(/"/g, '\\"');
    link.short_urls.forEach(shortUrl => {
      allRedirects.push(`    location = /${shortUrl} { return 301 "${url}"; }`);
    });
  }

  // Add sub-link redirects (now arrays)
  if (link.sub_links) {
    link.sub_links.forEach(subLink => {
      if (subLink.short_urls && subLink.short_urls.length > 0) {
        const url = subLink.url.replace(/"/g, '\\"');
        subLink.short_urls.forEach(shortUrl => {
          allRedirects.push(`    location = /${shortUrl} { return 301 "${url}"; }`);
        });
      }
    });
  }
});

const redirects = allRedirects.join('\n');

// Read the nginx config template
const nginxTemplate = fs.readFileSync('nginx.conf.template', 'utf8');

// Replace the placeholder with actual redirects
const nginxConfig = nginxTemplate.replace('    # SHORTLINK_REDIRECTS_PLACEHOLDER', redirects);

// Write the final nginx.conf
fs.writeFileSync('nginx.conf', nginxConfig);

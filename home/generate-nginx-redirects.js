#!/usr/bin/env node

import fs from 'fs';
import yaml from 'js-yaml';

const linksConfig = yaml.load(fs.readFileSync('links.yaml', 'utf8'));

const allRedirects = [];
const allItems = [];
linksConfig.links.forEach(link => {
  if (link.short_urls && link.url) {
    allItems.push({ url: link.url, short_urls: link.short_urls });
  }
  if (link.sub_links) {
    link.sub_links.forEach(subLink => {
      if (subLink.short_urls && subLink.url) {
        allItems.push({ url: subLink.url, short_urls: subLink.short_urls });
      }
    });
  }
});

const version = Date.now().toString(36);

allItems.forEach(item => {
  item.short_urls.forEach(shortUrl => {
    const hasUsername = item.url.includes('$username');

    const lines = [
      `    location = /${shortUrl} {`,
      `        set $target_url "${item.url}";`,
      `        add_header X-Redirect-Version "${version}";`,
      hasUsername && `        add_header Vary "X-Auth-Request-Email";`,
      `        return 301 $target_url;`,
      `    }`
    ].filter(Boolean);

    allRedirects.push(lines.join('\n'));
  });
});

const redirects = allRedirects.join('\n');

// Read the nginx config template
const nginxTemplate = fs.readFileSync('nginx.conf.template', 'utf8');

// Replace placeholders with actual values
let nginxConfig = nginxTemplate.replace('    # SHORTLINK_REDIRECTS_PLACEHOLDER', redirects);
nginxConfig = nginxConfig.replace('BUILD_VERSION_PLACEHOLDER', version);

// Write the final nginx.conf
fs.writeFileSync('nginx.conf', nginxConfig);

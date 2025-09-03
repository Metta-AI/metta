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

allItems.forEach(item => {
  item.short_urls.forEach(shortUrl => {
    // Check if URL contains $username variable
    if (item.url.includes('$username')) {
      // For URLs with username, build the URL dynamically
      const urlParts = item.url.split('$username');
      allRedirects.push(`    location = /${shortUrl} {
        # Build URL with username substitution
        set $target_url "${urlParts[0]}$username${urlParts[1] || ''}";
        return 302 $target_url;
    }`);
    } else {
      // Simple redirect for URLs without username
      allRedirects.push(`    location = /${shortUrl} { return 301 "${item.url}"; }`);
    }
  });
});

const redirects = allRedirects.join('\n');

// Read the nginx config template
const nginxTemplate = fs.readFileSync('nginx.conf.template', 'utf8');

// Replace the placeholder with actual redirects
const nginxConfig = nginxTemplate.replace('    # SHORTLINK_REDIRECTS_PLACEHOLDER', redirects);

// Write the final nginx.conf
fs.writeFileSync('nginx.conf', nginxConfig);

#!/usr/bin/env node

const fs = require('fs');
const yaml = require('js-yaml');
const path = require('path');

// Read the links config
const linksConfig = yaml.load(fs.readFileSync('links.yaml', 'utf8'));

// Generate nginx location blocks for redirects
const redirects = linksConfig.links
  .filter(link => link.short_url)
  .map(link => {
    // Escape quotes in URLs
    const url = link.url.replace(/"/g, '\\"');
    return `    location = /${link.short_url} { return 301 "${url}"; }`;
  })
  .join('\n');

// Read the nginx config template
const nginxTemplate = fs.readFileSync('nginx.conf.template', 'utf8');

// Replace the placeholder with actual redirects
const nginxConfig = nginxTemplate.replace('    # SHORTLINK_REDIRECTS_PLACEHOLDER', redirects);

// Write the final nginx.conf
fs.writeFileSync('nginx.conf', nginxConfig);

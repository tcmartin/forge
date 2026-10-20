#!/usr/bin/env node
'use strict';

const https = require('https');
const fs = require('fs');
const crypto = require('crypto');
const path = require('path');
const { pipeline, Transform } = require('stream');
const { execFileSync } = require('child_process');
const native = require('./native');

const pkg = require('./package.json');
const VERSION = pkg.version;
const APP = 'forge';

function validatedRepository() {
  const repository = (process.env.FORGE_RELEASE_REPO || '').trim();
  if (!/^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/.test(repository)) {
    throw new Error('Set FORGE_BINARY_PATH to a verified local binary or FORGE_RELEASE_REPO to an explicit owner/repository.');
  }
  return repository;
}

function download(url, destination, redirects = 0) {
  if (redirects > 5) return Promise.reject(new Error('Too many redirects'));
  return new Promise((resolve, reject) => {
    let settled = false;
    let timer;
    const finish = (error) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      if (error) reject(error); else resolve();
    };
    const request = https.get(url, (response) => {
      if ([301, 302, 307, 308].includes(response.statusCode)) {
        const location = response.headers.location;
        response.resume();
        clearTimeout(timer);
        if (!location || !location.startsWith('https://')) {
          finish(new Error('Release redirect must use HTTPS'));
          return;
        }
        download(location, destination, redirects + 1).then(() => finish()).catch(finish);
        return;
      }
      if (response.statusCode !== 200) {
        response.resume();
        finish(new Error(`HTTP ${response.statusCode} downloading ${url}`));
        return;
      }
      let bytes = 0;
      const bound = new Transform({ transform(chunk, _encoding, callback) {
        bytes += chunk.length;
        callback(bytes > 128 * 1024 * 1024 ? new Error('Release download exceeds byte limit') : null, chunk);
      } });
      pipeline(response, bound, fs.createWriteStream(destination, { flags: 'wx' }), finish);
    });
    request.on('error', finish);
    timer = setTimeout(() => request.destroy(new Error('Release download deadline exceeded')), 30000);
  });
}

function hashFile(filename) {
  return new Promise((resolve, reject) => {
    const hash = crypto.createHash('sha256');
    const input = fs.createReadStream(filename);
    input.on('data', (chunk) => hash.update(chunk));
    input.on('end', () => resolve(hash.digest('hex')));
    input.on('error', reject);
  });
}

async function main() {
  const platform = native.getPlatform();
  const { target, ext, archive } = platform;
  const local = process.env.FORGE_BINARY_PATH;
  const repository = !local && process.env.FORGE_RELEASE_REPO ? validatedRepository() : null;
  const installed = native.installedBinary(platform);
  const requestedSource = repository ? `release:${repository}` : null;
  if (!local && installed && (!requestedSource || installed.record.source === requestedSource)
      && await hashFile(installed.binary) === installed.record.sha256) {
    console.log(`forge: verified installed v${VERSION} binary.`);
    return;
  }
  // Missing/stale manifests never authorize reuse, including scripts-disabled upgrades.
  if (!local && !repository) validatedRepository();
  fs.mkdirSync(native.directory, { recursive: true });
  const staging = fs.mkdtempSync(path.join(native.directory, '.install-'));
  const destination = path.join(native.directory, `${APP}${ext}`);
  const stagedBinary = path.join(staging, `${APP}${ext}`);
  const archiveName = `${APP}-${target}${archive}`;
  const temporary = path.join(staging, `archive${archive}`);
  const sums = path.join(staging, 'SHA256SUMS');
  try {
    if (local) {
      const resolved = path.resolve(local);
      if (!fs.statSync(resolved).isFile()) throw new Error(`FORGE_BINARY_PATH is not a file: ${resolved}`);
      fs.copyFileSync(resolved, stagedBinary);
    } else {
      const url = `https://github.com/${repository}/releases/download/v${VERSION}/${archiveName}`;
      console.log(`forge: downloading v${VERSION} for ${process.platform}/${process.arch}`);
      await download(url, temporary);
      await download(`https://github.com/${repository}/releases/download/v${VERSION}/SHA256SUMS`, sums);
      const checksumLine = fs.readFileSync(sums, 'utf8')
        .split(/\r?\n/)
        .map((line) => line.trim().split(/\s+/, 2))
        .find((parts) => parts.length === 2 && parts[1].replace(/^\*/, '') === archiveName);
      if (!checksumLine) throw new Error(`SHA256SUMS does not list ${archiveName}`);
      const actual = await hashFile(temporary);
      if (actual.toLowerCase() !== checksumLine[0].toLowerCase()) {
        throw new Error(`Checksum verification failed for ${archiveName}`);
      }
      if (archive === '.zip') {
        execFileSync('powershell', ['-NoProfile', '-NonInteractive', '-Command', `Expand-Archive -Force -LiteralPath '${temporary.replace(/'/g, "''")}' -DestinationPath '${staging.replace(/'/g, "''")}'`]);
      } else {
        execFileSync('tar', ['-xzf', temporary, '-C', staging, '--', `${APP}${ext}`]);
      }
    }
    if (!fs.lstatSync(stagedBinary).isFile()) throw new Error(`Install source did not contain a regular ${APP}${ext}`);
    if (ext === '') fs.chmodSync(stagedBinary, 0o755);
    const sha256 = await hashFile(stagedBinary);
    const stat = fs.statSync(stagedBinary);
    const record = { schemaVersion: 1, packageVersion: VERSION, target,
      source: local ? 'local' : requestedSource, sha256,
      size: stat.size, mtimeMs: stat.mtimeMs, dev: stat.dev, ino: stat.ino };
    const stagedManifest = path.join(staging, 'manifest.json');
    fs.writeFileSync(stagedManifest, JSON.stringify(record) + '\n', { flag: 'wx' });
    // Publish complete binary bytes first. A stale manifest will reject a partial upgrade.
    fs.renameSync(stagedBinary, destination);
    fs.renameSync(stagedManifest, native.manifest);
  } finally {
    fs.rmSync(staging, { recursive: true, force: true });
  }
  console.log('forge: ready — run `forge` to start.');
}

main().catch((error) => {
  console.error(`\nforge install failed: ${error.message}\n`);
  process.exit(1);
});

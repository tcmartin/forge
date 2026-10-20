'use strict';

const fs = require('fs');
const path = require('path');
const pkg = require('./package.json');

const directory = path.join(__dirname, 'native');
const manifest = path.join(directory, 'manifest.json');

function getPlatform() {
  const supported = new Map([
    ['win32/x64', { target: 'windows-x86_64', ext: '.exe', archive: '.zip' }],
    ['linux/x64', { target: 'linux-x86_64', ext: '', archive: '.tar.gz' }],
    ['linux/arm64', { target: 'linux-aarch64', ext: '', archive: '.tar.gz' }],
    ['darwin/x64', { target: 'macos-x86_64', ext: '', archive: '.tar.gz' }],
    ['darwin/arm64', { target: 'macos-aarch64', ext: '', archive: '.tar.gz' }],
  ]);
  const result = supported.get(`${process.platform}/${process.arch}`);
  if (!result) throw new Error(`Unsupported platform: ${process.platform}/${process.arch}`);
  return result;
}

function installedBinary(platform) {
  const binary = path.join(directory, `forge${platform.ext}`);
  try {
    // Small manifest/stat checks keep normal CLI startup independent of binary size.
    // The installer additionally verifies SHA-256 before reusing these bytes.
    const record = JSON.parse(fs.readFileSync(manifest, 'utf8'));
    const stat = fs.lstatSync(binary);
    if (record.schemaVersion !== 1 || record.packageVersion !== pkg.version
        || record.target !== platform.target || !/^[a-f0-9]{64}$/.test(record.sha256)
        || !(record.source === 'local' || /^release:[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/.test(record.source))
        || !stat.isFile() || record.size !== stat.size || record.mtimeMs !== stat.mtimeMs
        || record.dev !== stat.dev || record.ino !== stat.ino
        || (platform.ext === '' && (stat.mode & 0o111) === 0)) return null;
    return { binary, record };
  } catch (_) {
    return null;
  }
}

module.exports = { directory, manifest, getPlatform, installedBinary };

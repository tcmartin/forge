'use strict';

const assert = require('assert');
const crypto = require('crypto');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawnSync, execFileSync } = require('child_process');

const source = path.resolve(__dirname, '..');
let passed = 0;
let failed = 0;

function fixture(run) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'forge-npm-test-'));
  const directory = path.join(root, 'package');
  fs.mkdirSync(directory);
  for (const name of ['install.js', 'native.js', 'package.json', 'bin']) {
    if (fs.existsSync(path.join(source, name))) {
      fs.cpSync(path.join(source, name), path.join(directory, name), { recursive: true });
    }
  }
  const environment = { ...process.env };
  for (const key of ['FORGE_BINARY_PATH', 'FORGE_RELEASE_REPO', 'NODE_OPTIONS']) delete environment[key];
  const binaryName = process.platform === 'win32' ? 'forge.exe' : 'forge';
  const native = path.join(directory, 'native', binaryName);
  const manifest = path.join(directory, 'native', 'manifest.json');
  const command = (script, env = {}) => spawnSync(process.execPath, [path.join(directory, script)], {
    env: { ...environment, ...env }, encoding: 'utf8', timeout: 10000,
  });
  const makeBinary = (name, contents) => {
    const filename = path.join(root, name);
    fs.writeFileSync(filename, contents, { mode: 0o755 });
    return filename;
  };
  const install = (contents = 'binary-v1') => {
    const result = command('install.js', { FORGE_BINARY_PATH: makeBinary('input', contents) });
    assert.strictEqual(result.status, 0, result.stderr);
  };
  try { run({ root, directory, native, manifest, command, makeBinary, install }); }
  finally { fs.rmSync(root, { recursive: true, force: true }); }
}

function test(name, run) {
  try { fixture(run); passed++; console.log(`PASS ${name}`); }
  catch (error) { failed++; console.error(`FAIL ${name}: ${error.message}`); }
}

test('fresh local install records package, target and binary hash', ({ native, manifest, install }) => {
  install();
  const record = JSON.parse(fs.readFileSync(manifest, 'utf8'));
  assert.strictEqual(record.packageVersion, require('../package.json').version);
  assert.strictEqual(record.sha256, crypto.createHash('sha256').update(fs.readFileSync(native)).digest('hex'));
  assert.strictEqual(record.source, 'local');
  assert.strictEqual(record.schemaVersion, 1);
});

test('explicit local override replaces an existing native binary', ({ native, command, makeBinary, install }) => {
  install();
  const result = command('install.js', { FORGE_BINARY_PATH: makeBinary('replacement', 'binary-v2') });
  assert.strictEqual(result.status, 0, result.stderr);
  assert.strictEqual(fs.readFileSync(native, 'utf8'), 'binary-v2');
});

test('repeat install reuses only a current verified native binary', ({ native, manifest, command, install }) => {
  install();
  const before = fs.statSync(native).mtimeMs;
  const result = command('install.js');
  assert.strictEqual(result.status, 0, result.stderr);
  assert.strictEqual(fs.statSync(native).mtimeMs, before);
  assert(fs.existsSync(manifest));
});

test('package upgrade rejects stale launcher and requires reinstall', ({ directory, native, command, install }) => {
  install();
  const pkg = JSON.parse(fs.readFileSync(path.join(directory, 'package.json'), 'utf8'));
  pkg.version = '99.0.0';
  fs.writeFileSync(path.join(directory, 'package.json'), JSON.stringify(pkg));
  const launch = command('bin/forge');
  assert.notStrictEqual(launch.status, 0);
  assert.match(launch.stderr, /reinstall|Reinstall/);
  const reinstall = command('install.js');
  assert.notStrictEqual(reinstall.status, 0);
  assert.strictEqual(fs.readFileSync(native, 'utf8'), 'binary-v1');
  install('upgraded');
  assert.strictEqual(fs.readFileSync(native, 'utf8'), 'upgraded');
});

test('legacy native file without manifest is not accepted', ({ native, command }) => {
  fs.mkdirSync(path.dirname(native));
  fs.writeFileSync(native, 'legacy');
  assert.notStrictEqual(command('install.js').status, 0);
  const launch = command('bin/forge');
  assert.notStrictEqual(launch.status, 0);
  assert.match(launch.stderr, /reinstall|Reinstall/);
});

test('missing explicit override is not ignored when a cached binary exists', ({ native, root, command, install }) => {
  install();
  const result = command('install.js', { FORGE_BINARY_PATH: path.join(root, 'missing') });
  assert.notStrictEqual(result.status, 0);
  assert.strictEqual(fs.readFileSync(native, 'utf8'), 'binary-v1');
});

test('changed native bytes cannot be reused by installer', ({ native, command, install }) => {
  install();
  const stat = fs.statSync(native);
  fs.writeFileSync(native, 'binary-vX');
  fs.utimesSync(native, stat.atime, stat.mtime);
  assert.notStrictEqual(command('install.js').status, 0);
});

test('changed native size fails launcher before execution', ({ native, command, install }) => {
  install();
  fs.appendFileSync(native, 'altered');
  const result = command('bin/forge');
  assert.notStrictEqual(result.status, 0);
  assert.match(result.stderr, /reinstall|Reinstall/);
});

for (const [field, value] of [['schemaVersion', 999], ['target', 'foreign-target'], ['sha256', 'invalid']]) {
  test(`invalid ${field} manifest is not reusable`, ({ manifest, command, install }) => {
    install();
    const record = JSON.parse(fs.readFileSync(manifest, 'utf8'));
    record[field] = value;
    fs.writeFileSync(manifest, JSON.stringify(record));
    assert.notStrictEqual(command('install.js').status, 0);
    const launch = command('bin/forge');
    assert.notStrictEqual(launch.status, 0);
    assert.match(launch.stderr, /reinstall|Reinstall/);
  });
}

test('explicit invalid repository cannot silently reuse a local binary', ({ command, install }) => {
  install();
  const result = command('install.js', { FORGE_RELEASE_REPO: 'not-a-repository' });
  assert.notStrictEqual(result.status, 0);
});

test('explicit local override takes precedence over inherited repository', ({ native, command, makeBinary }) => {
  const result = command('install.js', {
    FORGE_BINARY_PATH: makeBinary('local', 'explicit-local'), FORGE_RELEASE_REPO: 'invalid',
  });
  assert.strictEqual(result.status, 0, result.stderr);
  assert.strictEqual(fs.readFileSync(native, 'utf8'), 'explicit-local');
});

test('all supported platform asset contracts are explicit', () => {
  const vm = require('vm');
  for (const [platform, arch, target, ext, archive] of [
    ['win32', 'x64', 'windows-x86_64', '.exe', '.zip'],
    ['linux', 'x64', 'linux-x86_64', '', '.tar.gz'],
    ['linux', 'arm64', 'linux-aarch64', '', '.tar.gz'],
    ['darwin', 'x64', 'macos-x86_64', '', '.tar.gz'],
    ['darwin', 'arm64', 'macos-aarch64', '', '.tar.gz'],
  ]) {
    const context = { require: require('module').createRequire(path.join(source, 'native.js')),
      __dirname: source, module: { exports: {} }, process: { platform, arch } };
    vm.runInNewContext(fs.readFileSync(path.join(source, 'native.js'), 'utf8'), context);
    assert.deepStrictEqual(JSON.parse(JSON.stringify(context.module.exports.getPlatform())), { target, ext, archive });
  }
});

function remoteFixture(root, member = 'forge', checksumValid = true) {
  const input = path.join(root, 'archive-input');
  fs.mkdirSync(input);
  fs.writeFileSync(path.join(input, member), 'release-binary');
  const archive = path.join(root, 'release.tar.gz');
  execFileSync('tar', ['-czf', archive, '-C', input, member], { timeout: 10000 });
  const digest = checksumValid ? crypto.createHash('sha256').update(fs.readFileSync(archive)).digest('hex') : '0'.repeat(64);
  const { target } = require('../native').getPlatform();
  const asset = `forge-${target}.tar.gz`;
  const sums = path.join(root, 'sums');
  fs.writeFileSync(sums, `${digest}  ${asset}\n`);
  const preload = path.join(root, 'https-fixture.cjs');
  // All download bytes come from local files. An unexpected request fails.
  fs.writeFileSync(preload, `
    const { EventEmitter } = require('events');
    const { Readable } = require('stream');
    require('https').get = (url, callback) => {
      const request = new EventEmitter();
      request.destroy = (error) => request.emit('error', error);
      process.nextTick(() => {
        const base = 'https://github.com/fixture/forge/releases/download/v${require('../package.json').version}/';
        let file;
        if (url === base + ${JSON.stringify(asset)}) file = ${JSON.stringify(archive)};
        else if (url === base + 'SHA256SUMS') file = ${JSON.stringify(sums)};
        else { request.emit('error', new Error('Unexpected release URL: ' + url)); return; }
        const response = Readable.from([require('fs').readFileSync(file)]);
        response.statusCode = 200;
        response.headers = {};
        callback(response);
      });
      return request;
    };
  `);
  return { FORGE_RELEASE_REPO: 'fixture/forge', NODE_OPTIONS: `--require=${preload}` };
}

if (process.platform !== 'win32') {
  test('release archive replaces local binary and records exact release source', ({ root, native, manifest, command, install }) => {
    install();
    const result = command('install.js', remoteFixture(root));
    assert.strictEqual(result.status, 0, result.stderr);
    assert.strictEqual(fs.readFileSync(native, 'utf8'), 'release-binary');
    assert.strictEqual(JSON.parse(fs.readFileSync(manifest, 'utf8')).source, 'release:fixture/forge');
    assert.strictEqual(command('install.js', { FORGE_RELEASE_REPO: 'fixture/forge' }).status, 0);
  });
  test('release checksum failure preserves existing verified installation', ({ root, native, manifest, command, install }) => {
    install();
    const before = fs.readFileSync(manifest, 'utf8');
    const result = command('install.js', remoteFixture(root, 'forge', false));
    assert.notStrictEqual(result.status, 0);
    assert.match(result.stderr, /Checksum verification failed/);
    assert.strictEqual(fs.readFileSync(native, 'utf8'), 'binary-v1');
    assert.strictEqual(fs.readFileSync(manifest, 'utf8'), before);
    assert.strictEqual(fs.readdirSync(path.dirname(native)).length, 2);
  });
  test('wrong release member cannot bless the old destination as current', ({ root, native, manifest, command, install }) => {
    install();
    const before = fs.readFileSync(manifest, 'utf8');
    const result = command('install.js', remoteFixture(root, 'claurst'));
    assert.notStrictEqual(result.status, 0);
    assert.strictEqual(fs.readFileSync(native, 'utf8'), 'binary-v1');
    assert.strictEqual(fs.readFileSync(manifest, 'utf8'), before);
    assert.strictEqual(fs.readdirSync(path.dirname(native)).length, 2);
  });
  test('valid launcher forwards arguments and exit status', ({ root, native, command, makeBinary }) => {
    const input = makeBinary('executable', '#!/bin/sh\nprintf "forge-fixture\\n"\nexit 7\n');
    const install = command('install.js', { FORGE_BINARY_PATH: input });
    assert.strictEqual(install.status, 0, install.stderr);
    const result = command('bin/forge');
    assert.strictEqual(result.status, 7, result.stderr);
    assert.strictEqual(result.stdout, 'forge-fixture\n');
    assert.strictEqual(fs.statSync(native).mode & 0o111, 0o111);
    assert.strictEqual(fs.readdirSync(path.join(root, 'package', 'native')).length, 2);
  });
}

console.log(JSON.stringify({ passed, failed }));
process.exitCode = failed ? 1 : 0;

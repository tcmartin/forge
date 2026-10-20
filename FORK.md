# Forge

Forge is a developing Rust coding-agent fork of [Claurst](https://github.com/Kuberwastaken/claurst). The source baseline is `b0637c97ec34144387cbf2f74f65df6d16a6cef1`. The original GPL license and attribution are retained in [LICENSE.md](LICENSE.md) and [src-rust/LICENSE](src-rust/LICENSE). Distribution must retain the applicable GPL terms, attribution, and source availability.

The fork combines useful, independently specified Bullet behaviors with mechanisms studied in open-source agents. No Bullet implementation or proprietary prompts are included. Research specifications live in the adjacent `bullet-notes` folder. Pi (MIT), Codex (Apache-2.0), OpenCode (MIT), and Goose (Apache-2.0) are references. Crush's source-available implementation is not used as an open-source donor.

Development priorities are ordered tool execution, bounded independent work, exact evidence retrieval, explicit verification outcomes, reliable finite context, controlled child agents, and local diagnostics. Existing Claurst provider, terminal, session, skill and permission capabilities remain the starting point. Native workload evaluation and cross-platform verification are ongoing. This source checkpoint is not a completed release.

The independent development fork is [tcmartin/forge](https://github.com/tcmartin/forge). It is not affiliated with the upstream projects. No upstream PRs or published releases have been created.

The product command, local npm wrapper, installers, editor extension, and ACP registry template use the Forge name. Internal Rust package names retain the `claurst*` prefix for compatibility. Local binary installation is the verified distribution path. Remote installers require an explicitly supplied repository and never fall back to an upstream release location. The placeholder ACP registry URLs use the reserved `.invalid` domain until an authorized release repository exists.

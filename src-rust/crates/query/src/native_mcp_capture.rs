//! Bounded, source-backed MCP diagnostic projection for P07.
//!
//! This module is additive diagnostic data. It observes the checked MCP
//! snapshot and the query-owned active schema after receipt activation. It does
//! not read or reconstruct RMCP `tools/list` transport bytes.

use claurst_core::crypto_utils::sha256_hex;
use claurst_tools::{CombinedCatalogSnapshot, DiscoveryReceipt, McpManager};
use serde::Serialize;
use serde_json::json;
use std::collections::BTreeSet;
use std::io::{self, Write};

use crate::active_schema::ActiveSchemaSet;

pub(crate) const MAX_DISCOVERY_EVENT_BYTES: usize = 1024 * 1024;
const MAX_DESCRIPTORS: usize = 4096;
const MAX_SELECTED: usize = 64;
const MAX_ACTIVE_NAMES: usize = 128;

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct NativeMcpCatalogHeader {
    pub server_name: String,
    pub source_identity: String,
    pub connection_generation: u64,
    pub catalog_generation: u64,
    pub tool_count: usize,
    pub definition_bytes: usize,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct NativeMcpDescriptorIdentity {
    pub server_name: String,
    pub raw_name: String,
    pub callable_name: String,
    pub source_identity: String,
    pub connection_generation: u64,
    pub catalog_generation: u64,
    pub definition_digest: String,
    pub definition_bytes: usize,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct NativeMcpSelectedInvocation {
    pub server_name: String,
    pub raw_name: String,
    pub callable_name: String,
    pub source_identity: String,
    pub connection_generation: u64,
    pub catalog_generation: u64,
    pub definition_digest: String,
    pub active_revision: u64,
    pub active_schema_digest: String,
}

/// Immutable identity captured at receipt activation and reused by every
/// later diagnostic invocation. A post-activation name lookup cannot create
/// this binding.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct NativeMcpActivationBinding {
    pub catalog_snapshot_sha256: String,
    pub catalog_id: u64,
    pub active_revision: u64,
    pub active_schema_digest: String,
    pub active_names: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct NativeMcpDiscoveryWitness {
    pub observation_id: String,
    pub source: &'static str,
    pub catalog_id: u64,
    pub catalog_snapshot_sha256: String,
    pub catalog_headers: Vec<NativeMcpCatalogHeader>,
    pub descriptors: Vec<NativeMcpDescriptorIdentity>,
    pub catalog_projection_hex: String,
    pub catalog_projection_sha256: String,
    pub active_revision: u64,
    pub active_schema_digest: String,
    pub active_schema_bytes_hex: String,
    pub active_schema_bytes_sha256: String,
    pub selected: Vec<NativeMcpSelectedInvocation>,
}

struct BoundedJsonWriter {
    bytes: Vec<u8>,
    limit: usize,
}

impl BoundedJsonWriter {
    fn new(limit: usize) -> Self {
        Self {
            bytes: Vec::with_capacity(limit.min(4096)),
            limit,
        }
    }
}

impl Write for BoundedJsonWriter {
    fn write(&mut self, chunk: &[u8]) -> io::Result<usize> {
        let next = self
            .bytes
            .len()
            .checked_add(chunk.len())
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "JSON bound overflow"))?;
        if next > self.limit {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "serialized MCP diagnostic exceeds bound",
            ));
        }
        self.bytes.extend_from_slice(chunk);
        Ok(chunk.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn bounded_json<T: Serialize>(value: &T, label: &str) -> Result<Vec<u8>, String> {
    let mut writer = BoundedJsonWriter::new(MAX_DISCOVERY_EVENT_BYTES);
    serde_json::to_writer(&mut writer, value).map_err(|error| format!("{label}: {error}"))?;
    Ok(writer.bytes)
}

fn hex_encode(raw: &[u8], label: &str) -> Result<String, String> {
    if raw.len() > MAX_DISCOVERY_EVENT_BYTES {
        return Err(format!("{label} exceeds diagnostic bound"));
    }
    let capacity = raw
        .len()
        .checked_mul(2)
        .ok_or_else(|| format!("{label} hex length overflow"))?;
    let mut encoded = String::with_capacity(capacity);
    for byte in raw {
        use std::fmt::Write as _;
        write!(&mut encoded, "{byte:02x}").map_err(|_| format!("{label} hex encoding failed"))?;
    }
    Ok(encoded)
}

fn checked_snapshot_shape(snapshot: &CombinedCatalogSnapshot) -> Result<(), String> {
    if snapshot.headers.len() > MAX_DESCRIPTORS || snapshot.descriptors.len() > MAX_DESCRIPTORS {
        return Err("checked_catalog_descriptor_bound_exceeded".to_string());
    }
    let mut estimated = 0usize;
    for header in &snapshot.headers {
        if header.tool_count > MAX_DESCRIPTORS
            || header.definition_bytes > MAX_DISCOVERY_EVENT_BYTES
        {
            return Err("checked_catalog_header_bound_exceeded".to_string());
        }
        estimated = estimated
            .checked_add(header.server_name.len())
            .and_then(|value| value.checked_add(header.source_identity.to_string().len()))
            .and_then(|value| value.checked_add(header.definition_bytes))
            .ok_or_else(|| "checked_catalog_shape_overflow".to_string())?;
    }
    for descriptor in &snapshot.descriptors {
        if descriptor.definition_bytes() > MAX_DISCOVERY_EVENT_BYTES {
            return Err("checked_catalog_definition_bound_exceeded".to_string());
        }
        estimated = estimated
            .checked_add(descriptor.server_name().len())
            .and_then(|value| value.checked_add(descriptor.raw_name().len()))
            .and_then(|value| value.checked_add(descriptor.callable_name().len()))
            .and_then(|value| value.checked_add(descriptor.source_identity().to_string().len()))
            .and_then(|value| value.checked_add(descriptor.definition_digest().len()))
            .and_then(|value| value.checked_add(descriptor.definition_bytes()))
            .ok_or_else(|| "checked_catalog_shape_overflow".to_string())?;
    }
    if estimated > MAX_DISCOVERY_EVENT_BYTES {
        return Err("checked_catalog_shape_exceeds_bound".to_string());
    }
    Ok(())
}

fn snapshot_projection(snapshot: &CombinedCatalogSnapshot) -> serde_json::Value {
    let headers = snapshot
        .headers
        .iter()
        .map(|header| {
            json!({
                "server_name": header.server_name,
                "source_identity": header.source_identity.to_string(),
                "connection_generation": header.connection_generation,
                "catalog_generation": header.catalog_generation,
                "tool_count": header.tool_count,
                "definition_bytes": header.definition_bytes,
            })
        })
        .collect::<Vec<_>>();
    let descriptors = snapshot
        .descriptors
        .iter()
        .map(|descriptor| {
            json!({
                "server_name": descriptor.server_name(),
                "raw_name": descriptor.raw_name(),
                "callable_name": descriptor.callable_name(),
                "source_identity": descriptor.source_identity().to_string(),
                "connection_generation": descriptor.connection_generation(),
                "catalog_generation": descriptor.catalog_generation(),
                "definition_digest": descriptor.definition_digest(),
                "definition_bytes": descriptor.definition_bytes(),
            })
        })
        .collect::<Vec<_>>();
    json!({"format": "mcp_checked_descriptor_projection.v2", "headers": headers, "descriptors": descriptors})
}

pub(crate) fn checked_snapshot_digest(
    snapshot: &CombinedCatalogSnapshot,
) -> Result<String, String> {
    checked_snapshot_shape(snapshot)?;
    let bytes = bounded_json(&snapshot_projection(snapshot), "catalog projection")?;
    Ok(sha256_hex(&bytes))
}

pub(crate) fn activation_binding_from_snapshot(
    snapshot: &CombinedCatalogSnapshot,
    active: &ActiveSchemaSet,
) -> Result<NativeMcpActivationBinding, String> {
    let digest = checked_snapshot_digest(snapshot)?;
    let mut active_names = active
        .definitions()
        .map(|definition| definition.name.clone())
        .collect::<Vec<_>>();
    if active_names.len() > MAX_ACTIVE_NAMES {
        return Err("active_schema_name_bound_exceeded".to_string());
    }
    active_names.sort();
    Ok(NativeMcpActivationBinding {
        catalog_snapshot_sha256: digest,
        catalog_id: active.catalog_id(),
        active_revision: active.active_revision(),
        active_schema_digest: active.digest().to_string(),
        active_names,
    })
}

pub(crate) fn activation_binding(
    manager: Option<&McpManager>,
    active: &ActiveSchemaSet,
) -> Result<Option<NativeMcpActivationBinding>, String> {
    let Some(manager) = manager else {
        return Ok(None);
    };
    let snapshot = manager
        .checked_catalog_snapshot()
        .map_err(|error| error.to_string())?;
    Ok(Some(activation_binding_from_snapshot(&snapshot, active)?))
}

pub(crate) fn build_discovery_witness(
    manager: Option<&McpManager>,
    receipts: &[DiscoveryReceipt],
    active: &ActiveSchemaSet,
    activation: &NativeMcpActivationBinding,
) -> Result<Option<NativeMcpDiscoveryWitness>, String> {
    let Some(manager) = manager else {
        return Ok(None);
    };
    let snapshot = manager
        .checked_catalog_snapshot()
        .map_err(|error| error.to_string())?;
    checked_snapshot_shape(&snapshot)?;
    let snapshot_digest = checked_snapshot_digest(&snapshot)?;
    let mut current_names = active
        .definitions()
        .map(|definition| definition.name.clone())
        .collect::<Vec<_>>();
    current_names.sort();
    if snapshot_digest != activation.catalog_snapshot_sha256
        || active.catalog_id() != activation.catalog_id
        || active.active_revision() != activation.active_revision
        || active.digest() != activation.active_schema_digest
        || current_names != activation.active_names
    {
        return Err("mcp_activation_binding_changed".to_string());
    }
    let headers = snapshot
        .headers
        .iter()
        .map(|header| NativeMcpCatalogHeader {
            server_name: header.server_name.clone(),
            source_identity: header.source_identity.to_string(),
            connection_generation: header.connection_generation,
            catalog_generation: header.catalog_generation,
            tool_count: header.tool_count,
            definition_bytes: header.definition_bytes,
        })
        .collect::<Vec<_>>();
    let descriptors = snapshot
        .descriptors
        .iter()
        .map(|descriptor| NativeMcpDescriptorIdentity {
            server_name: descriptor.server_name().to_string(),
            raw_name: descriptor.raw_name().to_string(),
            callable_name: descriptor.callable_name().to_string(),
            source_identity: descriptor.source_identity().to_string(),
            connection_generation: descriptor.connection_generation(),
            catalog_generation: descriptor.catalog_generation(),
            definition_digest: descriptor.definition_digest().to_string(),
            definition_bytes: descriptor.definition_bytes(),
        })
        .collect::<Vec<_>>();
    let catalog_projection = bounded_json(&snapshot_projection(&snapshot), "catalog projection")?;
    let definitions = active.definitions().collect::<Vec<_>>();
    let active_schema_bytes = bounded_json(&definitions, "active schema")?;
    let active_names = activation.active_names.iter().collect::<BTreeSet<_>>();
    let mut selected = Vec::new();
    for receipt in receipts {
        for name in receipt.selected_names() {
            if !active_names.contains(&name) {
                return Err("receipt_selection_missing_from_activation_schema".to_string());
            }
            if let Some(descriptor) = snapshot
                .descriptors
                .iter()
                .find(|descriptor| descriptor.callable_name() == name)
            {
                selected.push(NativeMcpSelectedInvocation {
                    server_name: descriptor.server_name().to_string(),
                    raw_name: descriptor.raw_name().to_string(),
                    callable_name: descriptor.callable_name().to_string(),
                    source_identity: descriptor.source_identity().to_string(),
                    connection_generation: descriptor.connection_generation(),
                    catalog_generation: descriptor.catalog_generation(),
                    definition_digest: descriptor.definition_digest().to_string(),
                    active_revision: activation.active_revision,
                    active_schema_digest: activation.active_schema_digest.clone(),
                });
            } else {
                return Err("receipt_selection_missing_from_checked_snapshot".to_string());
            }
        }
    }
    if selected.is_empty() {
        return Ok(None);
    }
    if selected.len() > MAX_SELECTED {
        return Err("selected_invocation_bound_exceeded".to_string());
    }
    selected.sort_by(|left, right| {
        (
            &left.server_name,
            &left.callable_name,
            &left.source_identity,
            left.connection_generation,
            left.catalog_generation,
        )
            .cmp(&(
                &right.server_name,
                &right.callable_name,
                &right.source_identity,
                right.connection_generation,
                right.catalog_generation,
            ))
    });
    selected.dedup();
    let witness = NativeMcpDiscoveryWitness {
        observation_id: uuid::Uuid::new_v4().to_string(),
        source: "mcp_manager_checked_snapshot",
        catalog_id: activation.catalog_id,
        catalog_snapshot_sha256: snapshot_digest,
        catalog_headers: headers,
        descriptors,
        catalog_projection_hex: hex_encode(&catalog_projection, "catalog projection")?,
        catalog_projection_sha256: sha256_hex(&catalog_projection),
        active_revision: activation.active_revision,
        active_schema_digest: activation.active_schema_digest.clone(),
        active_schema_bytes_hex: hex_encode(&active_schema_bytes, "active schema")?,
        active_schema_bytes_sha256: sha256_hex(&active_schema_bytes),
        selected,
    };
    let _ = bounded_json(&witness, "mcp discovery envelope")?;
    Ok(Some(witness))
}

pub(crate) fn invocation_identity(
    manager: Option<&McpManager>,
    callable_name: &str,
    activation: &NativeMcpActivationBinding,
    active: &ActiveSchemaSet,
) -> Result<Option<NativeMcpSelectedInvocation>, String> {
    let Some(manager) = manager else {
        return Ok(None);
    };
    if !activation
        .active_names
        .iter()
        .any(|name| name == callable_name)
    {
        return Err("mcp_invocation_not_in_activation_schema".to_string());
    }
    let mut current_names = active
        .definitions()
        .map(|definition| definition.name.clone())
        .collect::<Vec<_>>();
    current_names.sort();
    if active.catalog_id() != activation.catalog_id
        || active.active_revision() != activation.active_revision
        || active.digest() != activation.active_schema_digest
        || current_names != activation.active_names
    {
        return Err("active_schema_binding_changed_after_activation".to_string());
    }
    let snapshot = manager
        .checked_catalog_snapshot()
        .map_err(|error| error.to_string())?;
    if checked_snapshot_digest(&snapshot)? != activation.catalog_snapshot_sha256 {
        return Err("checked_catalog_snapshot_changed_after_activation".to_string());
    }
    let descriptor = snapshot
        .descriptors
        .iter()
        .find(|descriptor| descriptor.callable_name() == callable_name);
    let Some(descriptor) = descriptor else {
        return Ok(None);
    };
    if !active.is_disclosed(callable_name) {
        return Err("invocation_descriptor_not_in_current_schema".to_string());
    }
    Ok(Some(NativeMcpSelectedInvocation {
        server_name: descriptor.server_name().to_string(),
        raw_name: descriptor.raw_name().to_string(),
        callable_name: descriptor.callable_name().to_string(),
        source_identity: descriptor.source_identity().to_string(),
        connection_generation: descriptor.connection_generation(),
        catalog_generation: descriptor.catalog_generation(),
        definition_digest: descriptor.definition_digest().to_string(),
        active_revision: activation.active_revision,
        active_schema_digest: activation.active_schema_digest.clone(),
    }))
}

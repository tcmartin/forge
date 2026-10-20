//! Finite envelopes for auxiliary command inference.

use claurst_api::{MediaAccounting, ModelRegistry, WireBudget};
use claurst_core::config::Config;
use claurst_query::{BudgetSource, RequestBudget};

pub(super) fn command_wire_budget(
    config: &Config,
    provider: &str,
    model: &str,
    output_reserve: u64,
) -> Result<WireBudget, String> {
    let model = super::stripped_model_for_provider(provider, model);
    let providers = super::provider_lookup_ids(provider);
    let explicit = providers.iter().find_map(|provider| {
        config
            .model_overrides
            .get(&format!("{provider}/{model}"))
            .and_then(|entry| entry.context_window)
            .map(u64::from)
    });
    let registry = ModelRegistry::new();
    let known = providers.iter().find_map(|provider| {
        registry
            .get(provider, model)
            .map(|entry| u64::from(entry.info.context_window))
    });
    let budget =
        RequestBudget::resolve(explicit, known, None, None, output_reserve, 8 * 1024 * 1024)
            .map_err(|error| error.to_string())?;
    let source = match budget.source {
        BudgetSource::ExplicitConfig => "explicit",
        BudgetSource::ProviderRegistry => "registry",
        BudgetSource::ProviderReported => "reported",
        BudgetSource::UnknownFallback => "fallback",
    };
    WireBudget::finite(
        budget.context_limit_tokens,
        budget.input_limit_tokens,
        budget.output_reserve_tokens,
        budget.wire_byte_limit,
        MediaAccounting::None,
    )
    .and_then(|budget| budget.with_source(source))
    .map_err(|error| error.to_string())
}

pub(super) fn fallback_notice(budget: &WireBudget) -> Option<String> {
    (budget.budget_source == "fallback").then(|| {
        format!(
            "Unknown model context limit; using a finite {}-token fallback. Configure modelOverrides to set this model's limit.",
            budget.context_limit_tokens
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use claurst_core::config::ModelOverride;

    #[test]
    fn unknown_model_has_finite_visible_fallback() {
        let budget = command_wire_budget(&Config::default(), "test", "unknown", 64).unwrap();
        assert_eq!(budget.context_limit_tokens, 32_000);
        assert_eq!(budget.input_limit_tokens, 31_936);
        assert_eq!(budget.budget_source, "fallback");
        assert!(fallback_notice(&budget).unwrap().contains("32000"));
        assert_eq!(budget.media, MediaAccounting::None);
    }

    #[test]
    fn explicit_pin_is_authoritative_and_invalid_pin_fails() {
        let mut config = Config::default();
        config.model_overrides.insert(
            "test/custom".into(),
            ModelOverride {
                context_window: Some(12_000),
                ..Default::default()
            },
        );
        let budget = command_wire_budget(&config, "test", "test/custom", 64).unwrap();
        assert_eq!(budget.context_limit_tokens, 12_000);
        assert_eq!(budget.budget_source, "explicit");
        assert!(fallback_notice(&budget).is_none());
        config
            .model_overrides
            .get_mut("test/custom")
            .unwrap()
            .context_window = Some(0);
        assert!(command_wire_budget(&config, "test", "custom", 64).is_err());
    }
}

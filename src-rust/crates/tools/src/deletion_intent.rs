//! Turn-scoped deletion intent for native Bash file-deletion commands.

use std::collections::BTreeSet;
use std::future::Future;
use std::path::Path;

tokio::task_local! {
    static ACTIVE_AUTHORIZATION: DeletionAuthorization;
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DeletionAuthorization {
    explicit_request: bool,
    target_basenames: BTreeSet<String>,
}

impl DeletionAuthorization {
    pub fn from_user_text(text: &str) -> Self {
        let intent_text = strip_markdown_code(text);
        Self {
            explicit_request: has_explicit_request(&intent_text),
            target_basenames: reference_tokens(text),
        }
    }

    fn permits(&self, targets: &[String]) -> bool {
        self.explicit_request
            && !targets.is_empty()
            && targets
                .iter()
                .all(|target| self.target_basenames.contains(target))
    }
}

pub async fn with_deletion_authorization<F>(
    authorization: DeletionAuthorization,
    future: F,
) -> F::Output
where
    F: Future,
{
    ACTIVE_AUTHORIZATION.scope(authorization, future).await
}

/// Reject native Bash deletion unless the current query turn carries an exact
/// user-derived authorization. This grants no execution permission: the Bash
/// tool still runs its existing permission handler after this check.
pub fn authorize_bash_command(command: &str) -> Result<(), String> {
    let targets = match deletion_targets(command) {
        DeletionTargets::None => return Ok(()),
        DeletionTargets::Invalid => {
            return Err(
                "Deletion command was not run because its literal targets could not be verified against the latest user request."
                    .to_string(),
            );
        }
        DeletionTargets::Literal(targets) => targets,
    };
    let authorized = ACTIVE_AUTHORIZATION
        .try_with(|authorization| authorization.permits(&targets))
        .unwrap_or(false);
    if authorized {
        Ok(())
    } else {
        Err(
            "Deletion command was not run because the latest user request did not explicitly authorize every named target."
                .to_string(),
        )
    }
}

#[derive(Debug, PartialEq, Eq)]
enum DeletionTargets {
    None,
    Invalid,
    Literal(Vec<String>),
}

fn deletion_targets(command: &str) -> DeletionTargets {
    let mut found = Vec::new();
    for segment in shell_segments(command) {
        let words = shell_words(&segment);
        let Some((command_index, executable)) = command_word(&words) else {
            continue;
        };
        if !matches!(executable.as_str(), "rm" | "unlink" | "rmdir") {
            continue;
        }
        let mut options = true;
        let mut segment_targets = Vec::new();
        for argument in words.iter().skip(command_index + 1) {
            if options && argument == "--" {
                options = false;
                continue;
            }
            if options && argument.starts_with('-') && argument != "-" {
                continue;
            }
            options = false;
            let Some(target) = literal_basename(argument) else {
                return DeletionTargets::Invalid;
            };
            segment_targets.push(target);
        }
        if segment_targets.is_empty() {
            return DeletionTargets::Invalid;
        }
        found.extend(segment_targets);
    }
    if found.is_empty() {
        DeletionTargets::None
    } else {
        DeletionTargets::Literal(found)
    }
}

fn command_word(words: &[String]) -> Option<(usize, String)> {
    let mut index = 0;
    while words
        .get(index)
        .is_some_and(|word| is_environment_assignment(word))
    {
        index += 1;
    }
    while words
        .get(index)
        .is_some_and(|word| matches!(word.as_str(), "command" | "builtin"))
    {
        index += 1;
    }
    let word = words.get(index)?;
    let executable = Path::new(word)
        .file_name()?
        .to_string_lossy()
        .to_ascii_lowercase();
    Some((index, executable))
}

fn is_environment_assignment(word: &str) -> bool {
    let Some((name, _value)) = word.split_once('=') else {
        return false;
    };
    !name.is_empty()
        && name
            .chars()
            .all(|character| character == '_' || character.is_ascii_alphanumeric())
}

fn literal_basename(argument: &str) -> Option<String> {
    if argument.is_empty()
        || argument.chars().any(|character| {
            matches!(
                character,
                '$' | '*' | '?' | '[' | ']' | '{' | '}' | '`' | '<' | '>'
            )
        })
    {
        return None;
    }
    let trimmed = argument.trim_end_matches('/');
    let basename = Path::new(trimmed).file_name()?.to_string_lossy();
    if basename.is_empty() || basename == "." || basename == ".." {
        return None;
    }
    Some(basename.into_owned())
}

fn shell_segments(command: &str) -> Vec<String> {
    let mut segments = Vec::new();
    let mut current = String::new();
    let mut quote = None;
    let mut escaped = false;
    for character in command.chars() {
        if escaped {
            current.push(character);
            escaped = false;
            continue;
        }
        if character == '\\' && quote != Some('\'') {
            current.push(character);
            escaped = true;
            continue;
        }
        if matches!(character, '\'' | '"') {
            if quote == Some(character) {
                quote = None;
            } else if quote.is_none() {
                quote = Some(character);
            }
            current.push(character);
            continue;
        }
        if quote.is_none() && matches!(character, ';' | '|' | '&' | '\n') {
            if !current.trim().is_empty() {
                segments.push(std::mem::take(&mut current));
            } else {
                current.clear();
            }
        } else {
            current.push(character);
        }
    }
    if !current.trim().is_empty() {
        segments.push(current);
    }
    segments
}

fn shell_words(segment: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut current = String::new();
    let mut quote = None;
    let mut escaped = false;
    for character in segment.chars() {
        if escaped {
            current.push(character);
            escaped = false;
            continue;
        }
        if character == '\\' && quote != Some('\'') {
            escaped = true;
            continue;
        }
        if matches!(character, '\'' | '"') {
            if quote == Some(character) {
                quote = None;
            } else if quote.is_none() {
                quote = Some(character);
            } else {
                current.push(character);
            }
            continue;
        }
        if quote.is_none() && character.is_whitespace() {
            if !current.is_empty() {
                words.push(std::mem::take(&mut current));
            }
        } else {
            current.push(character);
        }
    }
    if !current.is_empty() {
        words.push(current);
    }
    words
}

fn strip_markdown_code(text: &str) -> String {
    let mut output = String::with_capacity(text.len());
    let mut characters = text.chars().peekable();
    let mut fence = false;
    let mut inline = false;
    while let Some(character) = characters.next() {
        if character == '`' {
            let mut ticks = 1;
            while characters.peek() == Some(&'`') {
                characters.next();
                ticks += 1;
            }
            if ticks >= 3 {
                fence = !fence;
                inline = false;
            } else if !fence {
                inline = !inline;
            }
            output.push(' ');
            continue;
        }
        if fence || inline {
            if character == '\n' {
                output.push('\n');
            }
        } else {
            output.push(character);
        }
    }
    output
}

fn has_explicit_request(text: &str) -> bool {
    text.split(['.', '!', '?', ';', '\n'])
        .any(|clause| clause_has_explicit_request(&word_tokens(clause)))
}

fn clause_has_explicit_request(words: &[String]) -> bool {
    words.iter().enumerate().any(|(index, word)| {
        if !matches!(
            word.as_str(),
            "delete" | "remove" | "unlink" | "rmdir" | "erase" | "purge" | "rm"
        ) {
            return false;
        }
        let preceding_clause = &words[..index];
        if preceding_clause.iter().rev().take(8).any(|word| {
            matches!(
                word.as_str(),
                "not" | "never" | "dont" | "without" | "avoid"
            )
        }) {
            return false;
        }
        if preceding_clause.iter().rev().take(5).any(|word| {
            matches!(
                word.as_str(),
                "explain" | "describe" | "discuss" | "meaning" | "means" | "what" | "why" | "how"
            )
        }) {
            return false;
        }
        let start = index.saturating_sub(3);
        let previous = &words[start..index];
        let framed = index == 0
            || previous.iter().any(|word| word == "please")
            || suffix(previous, &["can", "you"])
            || suffix(previous, &["could", "you"])
            || suffix(previous, &["would", "you"])
            || suffix(previous, &["will", "you"])
            || suffix(previous, &["want", "you", "to"])
            || suffix(previous, &["need", "you", "to"])
            || suffix(previous, &["go", "ahead", "and"])
            || suffix(previous, &["feel", "free", "to"])
            || suffix(previous, &["you", "may"])
            || previous
                .last()
                .is_some_and(|word| matches!(word.as_str(), "run" | "execute" | "then" | "and"));
        framed
    })
}

fn suffix(words: &[String], expected: &[&str]) -> bool {
    words.len() >= expected.len()
        && words[words.len() - expected.len()..]
            .iter()
            .map(String::as_str)
            .eq(expected.iter().copied())
}

fn word_tokens(text: &str) -> Vec<String> {
    text.split(|character: char| !character.is_alphanumeric() && character != '\'')
        .filter_map(|word| {
            let normalized = word.replace('\'', "").to_lowercase();
            (!normalized.is_empty()).then_some(normalized)
        })
        .collect()
}

fn reference_tokens(text: &str) -> BTreeSet<String> {
    let mut tokens = BTreeSet::new();
    for raw in text.split_whitespace() {
        let token = raw.trim_matches(|character: char| {
            matches!(
                character,
                '`' | '\''
                    | '"'
                    | ','
                    | ';'
                    | ':'
                    | '.'
                    | '!'
                    | '?'
                    | '('
                    | ')'
                    | '['
                    | ']'
                    | '{'
                    | '}'
            )
        });
        if token.is_empty() {
            continue;
        }
        tokens.insert(token.to_string());
        if let Some(basename) = Path::new(token).file_name() {
            tokens.insert(basename.to_string_lossy().into_owned());
        }
    }
    tokens
}

#[cfg(test)]
mod tests {
    use super::*;

    fn permits(user: &str, command: &str) -> bool {
        let authorization = DeletionAuthorization::from_user_text(user);
        let targets = match deletion_targets(command) {
            DeletionTargets::Literal(targets) => targets,
            _ => return false,
        };
        authorization.permits(&targets)
    }

    #[test]
    fn explicit_request_requires_every_literal_target() {
        assert!(permits(
            "Please delete alpha.txt and beta.txt.",
            "rm -f -- alpha.txt beta.txt"
        ));
        assert!(!permits(
            "Please delete alpha.txt.",
            "rm alpha.txt beta.txt"
        ));
        assert!(!permits("Please delete alpha.txt.", "rm different.txt"));
    }

    #[test]
    fn neutral_mentions_and_code_examples_are_not_authority() {
        assert!(!permits(
            "Inspect report.txt and tell me its length.",
            "rm report.txt"
        ));
        assert!(!permits("Explain `rm report.txt`.", "rm report.txt"));
        assert!(!permits(
            "What does this do?\n```sh\nrm report.txt\n```",
            "rm report.txt"
        ));
        assert!(!permits(
            "Explain rm report.txt semantics.",
            "rm report.txt"
        ));
    }

    #[test]
    fn negation_is_not_authority() {
        assert!(!permits("Do not delete report.txt.", "rm report.txt"));
        assert!(!permits("Never remove report.txt.", "unlink report.txt"));
        assert!(!permits("Avoid deleting report.txt.", "rm report.txt"));
        assert!(!permits(
            "I don't want you to delete report.txt.",
            "rm report.txt"
        ));
        assert!(!permits(
            "Please explain how to delete report.txt.",
            "rm report.txt"
        ));
    }

    #[test]
    fn target_paths_bind_by_exact_basename_not_stem() {
        assert!(permits(
            "Delete `/tmp/report.txt`.",
            "rm -- /tmp/report.txt"
        ));
        assert!(!permits("Delete report.", "rm -- /tmp/report.txt"));
        assert!(!permits("Delete Report.txt.", "rm -- /tmp/report.txt"));
    }

    #[test]
    fn detects_only_native_deletion_executables() {
        assert_eq!(
            deletion_targets("echo rm report.txt"),
            DeletionTargets::None
        );
        assert_eq!(
            deletion_targets("printf ok && /bin/rm -rf -- report.txt"),
            DeletionTargets::Literal(vec!["report.txt".to_string()])
        );
        assert_eq!(deletion_targets("rm '$TARGET'"), DeletionTargets::Invalid);
        assert_eq!(deletion_targets("rm *.tmp"), DeletionTargets::Invalid);
    }
}

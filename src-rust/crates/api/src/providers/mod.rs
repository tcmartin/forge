pub mod anthropic;
pub use anthropic::AnthropicProvider;

pub(crate) mod message_normalization;
pub(crate) mod request_options;

pub mod openai;
pub use openai::OpenAiProvider;

pub mod google;
pub use google::GoogleProvider;

pub mod minimax;
pub use minimax::MinimaxProvider;

pub mod openai_compat;
pub use openai_compat::OpenAiCompatProvider;

pub mod openai_compat_providers;
pub use openai_compat_providers::{
    baseten, cerebras, deepinfra, deepseek, fireworks, friendli, groq, huggingface, llama_cpp,
    lm_studio, mistral, moonshot, nebius, novita, nvidia, ollama, opencode_zen, openrouter,
    ovhcloud, perplexity, qwen, sambanova, scaleway, siliconflow, stepfun, together_ai, upstage,
    venice, vultr_ai, xai, zai, zhipu,
};

pub mod free;
pub use free::{catalog_entry, FreeEntry, FreeProvider, FreeUpstream, FREE_CATALOG};

pub mod cohere;
pub use cohere::CohereProvider;

pub mod azure;
pub use azure::AzureProvider;

pub mod bedrock;
pub use bedrock::BedrockProvider;

pub mod copilot;
pub use copilot::CopilotProvider;

pub mod codex;
pub use codex::CodexProvider;

#[cfg(test)]
async fn read_loopback_fixture_request(socket: &mut tokio::net::TcpStream) {
    use tokio::io::AsyncReadExt;
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        let mut request = Vec::new();
        loop {
            let mut bytes = [0_u8; 4096];
            let size = socket.read(&mut bytes).await.expect("fixture request read");
            assert!(size > 0, "fixture request closed before its body completed");
            request.extend_from_slice(&bytes[..size]);
            assert!(
                request.len() <= 256 * 1024,
                "fixture request exceeded bound"
            );
            let Some(header_end) = request.windows(4).position(|part| part == b"\r\n\r\n") else {
                continue;
            };
            let headers = std::str::from_utf8(&request[..header_end]).expect("fixture headers");
            let length = headers
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length").then(|| {
                        value
                            .trim()
                            .parse::<usize>()
                            .expect("fixture content length")
                    })
                })
                .expect("fixture content-length header");
            assert!(
                length <= 256 * 1024,
                "fixture content length exceeded bound"
            );
            if request.len() >= header_end + 4 + length {
                break;
            }
        }
    })
    .await
    .expect("fixture request timeout");
}

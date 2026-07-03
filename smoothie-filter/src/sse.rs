/// Result of feeding a chunk to the SSE parser.
pub struct SseParseResult {
    /// Number of new token-bearing data lines seen in this chunk.
    pub new_tokens: u64,
    /// If the terminal event contained `predicted_per_token_ms`, its value.
    pub terminal_timing_ms: Option<f64>,
    /// Whether `data: [DONE]` was seen, indicating end of stream.
    pub stream_done: bool,
}

/// Incremental SSE parser that handles arbitrary chunk boundaries.
///
/// Counts `data:` lines as token emissions without parsing their JSON,
/// and does one targeted parse on the terminal event for timing data.
pub struct SseParserState {
    partial_line: Vec<u8>,
    token_count: u64,
    first_token_seen: bool,
}

impl SseParserState {
    pub fn new() -> Self {
        Self {
            partial_line: Vec::new(),
            token_count: 0,
            first_token_seen: false,
        }
    }

    /// Feed a chunk of bytes from the response body.
    pub fn feed(&mut self, chunk: &[u8]) -> SseParseResult {
        let mut new_tokens: u64 = 0;
        let mut terminal_timing_ms: Option<f64> = None;
        let mut stream_done = false;
        let mut last_data_line: Option<Vec<u8>> = None;

        // Prepend any partial line from the previous chunk.
        let data = if self.partial_line.is_empty() {
            chunk.to_vec()
        } else {
            let mut combined = std::mem::take(&mut self.partial_line);
            combined.extend_from_slice(chunk);
            combined
        };

        let mut start = 0;
        for (i, &byte) in data.iter().enumerate() {
            if byte == b'\n' {
                let line = &data[start..i];
                self.process_line(line, &mut new_tokens, &mut stream_done, &mut last_data_line);
                start = i + 1;
            }
        }

        // Save any remaining bytes as a partial line.
        if start < data.len() {
            self.partial_line = data[start..].to_vec();
        }

        // On stream done, check the last data line for terminal timing.
        if stream_done
            && let Some(line_bytes) = last_data_line {
                terminal_timing_ms = Self::extract_terminal_timing(&line_bytes);
            }

        SseParseResult {
            new_tokens,
            terminal_timing_ms,
            stream_done,
        }
    }

    /// Total tokens observed across all feeds.
    pub fn token_count(&self) -> u64 {
        self.token_count
    }

    /// Whether the first token has been seen.
    pub fn first_token_seen(&self) -> bool {
        self.first_token_seen
    }

    fn process_line(
        &mut self,
        line: &[u8],
        new_tokens: &mut u64,
        stream_done: &mut bool,
        last_data_line: &mut Option<Vec<u8>>,
    ) {
        // Strip optional \r for \r\n line endings.
        let line = line.strip_suffix(b"\r").unwrap_or(line);

        if !line.starts_with(b"data: ") {
            return;
        }

        let payload = &line[6..];

        if payload == b"[DONE]" {
            *stream_done = true;
            return;
        }

        // This is a token-bearing data line.
        *last_data_line = Some(line.to_vec());
        self.token_count += 1;
        *new_tokens += 1;
        if !self.first_token_seen {
            self.first_token_seen = true;
        }
    }

    fn extract_terminal_timing(line_bytes: &[u8]) -> Option<f64> {
        // Only attempt JSON parse if the line contains the timing field.
        let line_str = std::str::from_utf8(line_bytes).ok()?;
        let payload = line_str.strip_prefix("data: ")?;

        if !payload.contains("predicted_per_token_ms") {
            return None;
        }

        // Targeted parse: extract just the timing value.
        let value: serde_json::Value = serde_json::from_str(payload).ok()?;
        value
            .get("timings")
            .and_then(|t| t.get("predicted_per_token_ms"))
            .and_then(|v| v.as_f64())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_simple_data_lines() {
        let mut parser = SseParserState::new();
        let chunk = b"data: {\"content\":\"hello\"}\n\ndata: {\"content\":\"world\"}\n\n";
        let result = parser.feed(chunk);
        assert_eq!(result.new_tokens, 2);
        assert!(!result.stream_done);
        assert_eq!(parser.token_count(), 2);
    }

    #[test]
    fn handles_done_marker() {
        let mut parser = SseParserState::new();
        let chunk = b"data: {\"content\":\"a\"}\n\ndata: [DONE]\n\n";
        let result = parser.feed(chunk);
        assert_eq!(result.new_tokens, 1);
        assert!(result.stream_done);
    }

    #[test]
    fn handles_split_chunks() {
        let mut parser = SseParserState::new();

        // First chunk ends mid-line.
        let result1 = parser.feed(b"data: {\"con");
        assert_eq!(result1.new_tokens, 0);

        // Second chunk completes the line.
        let result2 = parser.feed(b"tent\":\"a\"}\n\n");
        assert_eq!(result2.new_tokens, 1);
        assert_eq!(parser.token_count(), 1);
    }

    #[test]
    fn handles_crlf_line_endings() {
        let mut parser = SseParserState::new();
        let chunk = b"data: {\"content\":\"a\"}\r\n\r\ndata: [DONE]\r\n\r\n";
        let result = parser.feed(chunk);
        assert_eq!(result.new_tokens, 1);
        assert!(result.stream_done);
    }

    #[test]
    fn ignores_non_data_lines() {
        let mut parser = SseParserState::new();
        let chunk = b"event: message\nid: 1\ndata: {\"content\":\"a\"}\n\n";
        let result = parser.feed(chunk);
        assert_eq!(result.new_tokens, 1);
    }

    #[test]
    fn extracts_terminal_timing() {
        let mut parser = SseParserState::new();
        let chunk = b"data: {\"timings\":{\"predicted_per_token_ms\":42.5}}\n\ndata: [DONE]\n\n";
        let result = parser.feed(chunk);
        assert_eq!(result.new_tokens, 1);
        assert!(result.stream_done);
        assert!((result.terminal_timing_ms.unwrap() - 42.5).abs() < f64::EPSILON);
    }

    #[test]
    fn no_terminal_timing_without_field() {
        let mut parser = SseParserState::new();
        let chunk = b"data: {\"content\":\"last\"}\n\ndata: [DONE]\n\n";
        let result = parser.feed(chunk);
        assert!(result.stream_done);
        assert!(result.terminal_timing_ms.is_none());
    }

    #[test]
    fn multiple_feeds_accumulate_tokens() {
        let mut parser = SseParserState::new();
        parser.feed(b"data: {\"t\":1}\n\n");
        parser.feed(b"data: {\"t\":2}\n\n");
        parser.feed(b"data: {\"t\":3}\n\n");
        assert_eq!(parser.token_count(), 3);
        assert!(parser.first_token_seen());
    }

    #[test]
    fn empty_chunk_is_harmless() {
        let mut parser = SseParserState::new();
        let result = parser.feed(b"");
        assert_eq!(result.new_tokens, 0);
        assert!(!result.stream_done);
    }

    #[test]
    fn partial_line_across_three_chunks() {
        let mut parser = SseParserState::new();
        parser.feed(b"da");
        parser.feed(b"ta: {\"c\":\"");
        let result = parser.feed(b"x\"}\n\n");
        assert_eq!(parser.token_count(), 1);
        assert_eq!(result.new_tokens, 1);
    }
}

//! Simple word-based tokenizer for testing.
//! In production, use tiktoken or sentencepiece.

use std::collections::HashMap;

pub struct SimpleTokenizer {
    vocab: HashMap<String, usize>,
    reverse_vocab: HashMap<usize, String>,
}

impl SimpleTokenizer {
    pub fn new(vocab_size: usize) -> Self {
        let mut vocab = HashMap::new();
        let mut reverse_vocab = HashMap::new();

        // Reserved tokens
        vocab.insert("<pad>".to_string(), 0);
        vocab.insert("<unk>".to_string(), 1);
        vocab.insert("<eos>".to_string(), 2);

        reverse_vocab.insert(0, "<pad>".to_string());
        reverse_vocab.insert(1, "<unk>".to_string());
        reverse_vocab.insert(2, "<eos>".to_string());

        // Simple word vocab (for testing)
        let common_words = vec![
            "the", "a", "is", "in", "to", "of", "and", "for", "on", "with", "as", "at", "by",
            "from", "it", "this", "that", "are", "was", "be", "have", "has", "had", "not", "can",
            "will", "would", "could", "should", "said", "there", "their", "they", "we", "you",
            "he", "she", "I", "my", "your", "our", "but", "or", "so", "if", "then", "when",
            "where", "what", "how", "why", "who", "which", "do", "does", "did", "go", "come",
            "see", "get", "make", "know", "think", "take", "want", "use", "find", "give", "tell",
            "work", "call", "try", "ask", "need", "feel", "become", "leave", "put",
        ];

        for (i, word) in common_words.iter().enumerate() {
            let token_id = i + 3;
            if token_id >= vocab_size {
                break;
            }
            vocab.insert(word.to_string(), token_id);
            reverse_vocab.insert(token_id, word.to_string());
        }

        Self {
            vocab,
            reverse_vocab,
        }
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        text.split_whitespace()
            .map(|word| {
                *self.vocab.get(&word.to_lowercase()).unwrap_or(&1) // <unk>
            })
            .collect()
    }

    pub fn decode(&self, ids: &[usize]) -> String {
        ids.iter()
            .filter_map(|&id| self.reverse_vocab.get(&id))
            .cloned()
            .collect::<Vec<_>>()
            .join(" ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenizer_encode_decode() {
        let tok = SimpleTokenizer::new(1000);
        let text = "the quick brown fox";
        let ids = tok.encode(text);
        assert!(!ids.is_empty());

        let decoded = tok.decode(&ids);
        // Some words might be unknown, but common ones should work
        assert!(decoded.contains("the"));
    }
}

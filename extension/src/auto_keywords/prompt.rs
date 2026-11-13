//! The prompt sent with every photo, and the cleaning of the model's answer into one fixed
//! category and a short list of keywords.
//!
//! The model's text is untrusted input: a reply can arrive wrapped in explanations or Markdown
//! fences, name a category outside the fixed list, repeat itself, or smuggle a `|` into a keyword
//! — the hierarchical separator — to plant levels the user never asked for. Everything goes
//! through [`parse`] before a result is shown or written.

use std::collections::HashSet;

use serde_json::Value;

use lightcraft_catalog::keywords;

/// The fixed top-level categories, in display order (Simplified Chinese: the model is asked in
/// Chinese and the categories are the Chinese names themselves).
pub const CATEGORIES: [&str; 8] = ["人物", "动物", "风景", "建筑", "食物", "物体", "文档", "其他"];

/// The category for what fits nowhere else.
pub const OTHER: &str = "其他";

/// The most characters of one keyword: a rambling model answer must not become a paragraph.
pub const MAX_KEYWORD_CHARS: usize = 32;
pub const MAX_DESCRIPTION_CHARS: usize = 256;

/// The prompt sent with every photo (per §7.1 of the design). Sent fresh each time: no model-side
/// state is kept between photos.
pub fn prompt_text(max_keywords: usize) -> String {
    include_str!("prompt.txt")
        .trim()
        .replace("{categories}", &CATEGORIES.join("、"))
        .replace("{max_keywords}", &max_keywords.clamp(1, 16).to_string())
}

/// One photo's cleaned analysis.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Analysis {
    /// One of [`CATEGORIES`].
    pub category: String,
    /// Objective AI description; empty for older keyword-only services.
    pub description: String,
    /// Cleaned, deduplicated, at most `max_keywords` of them.
    pub keywords: Vec<String>,
}

/// What [`parse`] refuses: the reply held no JSON, or nothing usable survived the cleaning.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParseError {
    /// No JSON object or array was found in the reply.
    NoJson,
    /// There was JSON, but no keyword survived.
    NoKeywords,
}

/// The alias map for categories the model answered outside the fixed list (§7.2.3): the Chinese
/// name, its English counterpart, near spellings. Compared case-insensitively.
fn category_alias(name: &str) -> Option<&'static str> {
    let n = name.trim().to_lowercase();
    let aliases: [(&str, &str); 16] = [
        ("人像", "人物"),
        ("people", "人物"),
        ("person", "人物"),
        ("风光", "风景"),
        ("landscape", "风景"),
        ("nature", "风景"),
        ("architecture", "建筑"),
        ("美食", "食物"),
        ("food", "食物"),
        ("animal", "动物"),
        ("pet", "动物"),
        ("document", "文档"),
        ("物品", "物体"),
        ("product", "物体"),
        ("物件", "物体"),
        ("静物", "物体"),
    ];
    aliases.iter().find(|(from, _)| *from == n).map(|(_, to)| *to)
}

/// The canonical category for whatever the model named: a member of [`CATEGORIES`] stays, known
/// aliases map over, everything else lands on [`OTHER`].
fn canonical_category(name: &str) -> String {
    let name = name.trim();
    if CATEGORIES.iter().any(|c| keywords::same(c, name)) {
        return CATEGORIES.iter().find(|c| keywords::same(c, name)).map_or_else(|| name.to_string(), |c| (*c).to_string());
    }
    category_alias(name).unwrap_or(OTHER).to_string()
}

/// The JSON object or array inside a model reply: Markdown fences stripped, then the first `{`
/// to the last `}`, else the first `[` to the last `]`.
fn json_slice(text: &str) -> Option<&str> {
    let text = text.trim();
    // a fenced block is the reply's best candidate: take what is inside the outermost fence
    let fenced = |open: &str| {
        text.find(open).map(|start| {
            let after = &text[start + open.len()..];
            let end = after.find(open).map(|e| start + open.len() + e).unwrap_or(text.len());
            text[start + open.len()..end].trim()
        })
    };
    if let Some(fenced) = fenced("```") {
        return if fenced.starts_with('{') || fenced.starts_with('[') { Some(fenced) } else { json_slice(fenced) };
    }
    let (open, close) = if text.find('[').is_some_and(|a| text.find('{').is_none_or(|o| a < o)) { ('[', ']') } else { ('{', '}') };
    let first = text.find(open)?;
    let last = text.rfind(close)?;
    (first <= last).then(|| text[first..=last].trim())
}

/// The keyword list a JSON value offers: `keywords`, else `tags` or `labels` (models name the
/// field differently), else the value itself when it is an array or a single string.
fn keywords_of(v: &Value) -> Vec<String> {
    let from = |v: &Value| -> Vec<String> {
        match v {
            Value::Array(items) => items.iter().filter_map(|i| i.as_str().map(str::to_string)).collect(),
            Value::String(s) => vec![s.clone()],
            _ => Vec::new(),
        }
    };
    for key in ["keywords", "tags", "labels"] {
        if let Some(list) = v.get(key) {
            let list = from(list);
            if !list.is_empty() {
                return list;
            }
        }
    }
    from(v)
}

/// One cleaned keyword, or `None` when it has to go: empty after cleaning, the category or the
/// parent's own name, or a case-insensitive repeat of one already kept (`seen`).
fn clean_one(raw: &str, category: &str, parent: &str, seen: &mut HashSet<String>) -> Option<String> {
    let trimmed = raw.trim().strip_prefix('#').map(str::trim).unwrap_or_else(|| raw.trim());
    // `|` is the hierarchical separator: a model must never inject levels
    let flat: String = trimmed.chars().map(|c| if c == keywords::SEP || c.is_control() { ' ' } else { c }).collect();
    let kw: String = flat.trim().chars().take(MAX_KEYWORD_CHARS).collect();
    let kw = kw.trim();
    if kw.is_empty() || keywords::same(kw, category) || (!parent.is_empty() && keywords::same(kw, parent)) {
        return None;
    }
    if !seen.insert(kw.to_lowercase()) {
        return None;
    }
    Some(kw.to_string())
}

/// Parse and clean a model reply. Every keyword goes through [`clean_one`], which also drops the
/// parent's own name (`parent`, as written in the settings); at most `max_keywords` survive. An
/// answer with no usable keyword is [`ParseError::NoKeywords`] — one photo's failure, never a
/// stop for the batch.
pub fn parse(content: &str, max_keywords: usize, parent: &str) -> Result<Analysis, ParseError> {
    let Some(slice) = json_slice(content) else { return Err(ParseError::NoJson) };
    let Ok(v) = serde_json::from_str::<Value>(slice) else { return Err(ParseError::NoJson) };
    let (category, list) = match &v {
        Value::Array(_) => (OTHER.to_string(), keywords_of(&v)),
        Value::Object(_) => {
            let named = v.get("category").and_then(Value::as_str).unwrap_or(OTHER);
            (canonical_category(named), keywords_of(&v))
        }
        _ => return Err(ParseError::NoJson),
    };
    let max = max_keywords.max(1);
    let parent = keywords::clean(parent);
    let mut seen = HashSet::new();
    let keywords: Vec<String> = list.into_iter().filter_map(|kw| clean_one(&kw, &category, &parent, &mut seen)).take(max).collect();
    if keywords.is_empty() {
        return Err(ParseError::NoKeywords);
    }
    let raw = v.get("description").and_then(Value::as_str).unwrap_or_default();
    let description: String = raw.chars().map(|c| if c.is_control() { ' ' } else { c }).take(MAX_DESCRIPTION_CHARS).collect();
    Ok(Analysis { category, description: description.trim().to_string(), keywords })
}

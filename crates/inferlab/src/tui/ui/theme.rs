use crate::tui::{DisplayTone, State};
use ratatui::style::Color;

pub(super) const ACCENT: Color = Color::Rgb(10, 168, 232);
pub(super) const ACCENT_SOFT: Color = Color::Rgb(104, 146, 166);
pub(super) const MUTED: Color = Color::Rgb(102, 114, 124);
pub(super) const SECONDARY: Color = Color::Rgb(154, 166, 174);
pub(super) const SUCCESS: Color = Color::Rgb(92, 184, 132);
pub(super) const WARNING: Color = Color::Rgb(220, 174, 92);
pub(super) const CRITICAL: Color = Color::Rgb(224, 104, 104);
const INCOMPATIBLE: Color = Color::Rgb(188, 116, 204);

pub(super) fn tone_symbol(tone: DisplayTone) -> &'static str {
    match tone {
        DisplayTone::Normal => "·",
        DisplayTone::Success | DisplayTone::Active => "●",
        DisplayTone::Warning => "◆",
        DisplayTone::Critical => "×",
    }
}

pub(super) fn tone_color(tone: DisplayTone) -> Color {
    match tone {
        DisplayTone::Normal => MUTED,
        DisplayTone::Success => SUCCESS,
        DisplayTone::Active => ACCENT,
        DisplayTone::Warning => WARNING,
        DisplayTone::Critical => CRITICAL,
    }
}

pub(super) fn state_color(state: State) -> Color {
    match state {
        State::Live => SUCCESS,
        State::Stale => WARNING,
        State::Unavailable => CRITICAL,
        State::Incompatible => INCOMPATIBLE,
    }
}

pub(super) fn section_color(section: &str) -> Color {
    if section == "ATTENTION" {
        WARNING
    } else {
        ACCENT_SOFT
    }
}

#[cfg(test)]
mod tests {
    /// The TUI accent is the brand InferLab Blue; the website stylesheet owns
    /// the hex spelling (`--inferlab-blue`), so a brand refresh cannot drift
    /// the product surface silently. The SVG artworks stay artwork.
    #[test]
    fn accent_matches_the_brand_stylesheet() -> Result<(), Box<dyn std::error::Error>> {
        let brand = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../website/src/styles/brand.css"
        ))?;
        let ratatui::style::Color::Rgb(red, green, blue) = super::ACCENT else {
            return Err("the TUI accent is no longer a direct RGB color".into());
        };
        let hex = format!("#{red:02X}{green:02X}{blue:02X}");
        assert!(
            brand.contains(&format!("--inferlab-blue: {hex};")),
            "the TUI accent {hex} drifted from the brand stylesheet"
        );
        Ok(())
    }
}

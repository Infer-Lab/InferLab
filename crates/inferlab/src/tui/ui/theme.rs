//! The TUI's two palettes ([[ADR-0055]]): each is tuned for contrast on its
//! own background. Fills keep the brand blue; accent text takes each palette's
//! contrast-safe shade. Normal text uses the terminal's own foreground.

use crate::tui::appearance::Appearance;
use crate::tui::{DisplayTone, State};
use ratatui::style::{Color, Style};
use ratatui::widgets::{Block, BorderType};

/// The brand InferLab Blue, for fills only: selection bars, focus rings,
/// progress, and chips.
pub(super) const BRAND: Color = Color::Rgb(BRAND_RGB.0, BRAND_RGB.1, BRAND_RGB.2);
const BRAND_RGB: (u8, u8, u8) = (10, 168, 232);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::tui) struct Palette {
    pub(super) text: Color,
    pub(super) secondary: Color,
    pub(super) muted: Color,
    pub(super) faint: Color,
    pub(super) rule: Color,
    pub(super) accent: Color,
    pub(super) accent_soft: Color,
    pub(super) success: Color,
    pub(super) warning: Color,
    pub(super) critical: Color,
    pub(super) incompatible: Color,
    /// Text drawn on a saturated fill: a status pill.
    pub(super) on_fill: Color,
    /// Text drawn on a chip or the selection bar. A fill never leaves its
    /// text to the terminal's foreground, which may not contrast with it.
    pub(super) on_surface: Color,
    pub(super) chip: Color,
    pub(super) selection: Color,
    /// Panel and sidebar tints, only when the terminal reported its
    /// background: painting a tint over an unknown background would show as
    /// a patch.
    pub(super) panel: Option<Color>,
}

const DARK: Palette = Palette {
    text: Color::Reset,
    secondary: Color::Rgb(154, 166, 174),
    muted: Color::Rgb(118, 128, 138),
    faint: Color::Rgb(78, 88, 98),
    rule: Color::Rgb(58, 66, 74),
    accent: Color::Rgb(64, 182, 240),
    accent_soft: Color::Rgb(110, 160, 186),
    success: Color::Rgb(88, 190, 130),
    warning: Color::Rgb(226, 176, 80),
    critical: Color::Rgb(232, 104, 104),
    incompatible: Color::Rgb(188, 116, 204),
    on_fill: Color::Rgb(13, 17, 23),
    on_surface: Color::Rgb(220, 226, 232),
    chip: Color::Rgb(36, 44, 54),
    selection: Color::Rgb(20, 42, 58),
    panel: None,
};

const LIGHT: Palette = Palette {
    text: Color::Reset,
    secondary: Color::Rgb(72, 80, 88),
    muted: Color::Rgb(100, 108, 116),
    faint: Color::Rgb(160, 166, 172),
    rule: Color::Rgb(206, 210, 214),
    accent: Color::Rgb(0, 112, 170),
    accent_soft: Color::Rgb(50, 100, 130),
    success: Color::Rgb(24, 128, 72),
    warning: Color::Rgb(166, 104, 0),
    critical: Color::Rgb(190, 48, 48),
    incompatible: Color::Rgb(140, 60, 160),
    on_fill: Color::Rgb(255, 255, 255),
    on_surface: Color::Rgb(36, 42, 48),
    chip: Color::Rgb(228, 232, 236),
    selection: Color::Rgb(222, 238, 248),
    panel: None,
};

impl Default for Palette {
    fn default() -> Self {
        DARK
    }
}

impl Palette {
    pub(in crate::tui) fn for_appearance(appearance: Appearance) -> Self {
        let mut palette = if appearance.light { LIGHT } else { DARK };
        if let Some(background) = appearance.background {
            // A known background tints surfaces from itself, so the console
            // sits in the operator's own color scheme.
            let toward = if appearance.light {
                (0, 0, 0)
            } else {
                (255, 255, 255)
            };
            palette.panel = Some(blend(background, toward, 0.03));
            palette.chip = blend(background, toward, 0.12);
            palette.selection = blend(background, BRAND_RGB, 0.16);
        }
        palette
    }

    /// The selection bar, with its own text color.
    pub(super) fn selected(&self) -> Style {
        Style::default().bg(self.selection).fg(self.on_surface)
    }

    /// A keycap or chip: bold text on the chip fill.
    pub(super) fn keycap(&self) -> Style {
        Style::default()
            .fg(self.on_surface)
            .bg(self.chip)
            .add_modifier(ratatui::style::Modifier::BOLD)
    }

    pub(super) fn tone(&self, tone: DisplayTone) -> Color {
        match tone {
            DisplayTone::Normal => self.muted,
            DisplayTone::Success => self.success,
            DisplayTone::Active => self.accent,
            DisplayTone::Warning => self.warning,
            DisplayTone::Critical => self.critical,
        }
    }

    pub(super) fn state(&self, state: State) -> Color {
        match state {
            State::Live => self.success,
            State::Stale => self.warning,
            State::Unavailable => self.critical,
            State::Incompatible => self.incompatible,
        }
    }

    pub(super) fn section(&self, section: &str) -> Color {
        if section == "ATTENTION" {
            self.warning
        } else {
            self.accent_soft
        }
    }
}

/// A rounded panel whose border marks focus in the brand blue; it carries
/// the surface tint when the terminal reported its background.
pub(super) fn panel(p: Palette, focused: bool) -> Block<'static> {
    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(if focused { BRAND } else { p.rule }));
    match p.panel {
        Some(tint) => block.style(Style::default().bg(tint)),
        None => block,
    }
}

/// The frames an active operation's glyph cycles through.
pub(super) const SPINNER: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

/// A bar that fills `fraction` of `width` cells in eighth-cell steps, padded
/// to the full width.
pub(super) fn smooth_bar(fraction: f64, width: usize) -> (String, String) {
    const EIGHTHS: [&str; 8] = ["", "▏", "▎", "▍", "▌", "▋", "▊", "▉"];
    let eighths = (fraction.clamp(0.0, 1.0) * (width * 8) as f64).round() as usize;
    let mut filled = "█".repeat(eighths / 8);
    filled.push_str(EIGHTHS[eighths % 8]);
    let used = eighths / 8 + usize::from(eighths % 8 > 0);
    (filled, " ".repeat(width.saturating_sub(used)))
}

fn blend(from: (u8, u8, u8), toward: (u8, u8, u8), weight: f64) -> Color {
    let mix = |a: u8, b: u8| {
        let value = f64::from(a) + (f64::from(b) - f64::from(a)) * weight;
        // The weight keeps the mix inside 0..=255; the clamp makes the
        // conversion total.
        value.round().clamp(0.0, 255.0) as u8
    };
    Color::Rgb(
        mix(from.0, toward.0),
        mix(from.1, toward.1),
        mix(from.2, toward.2),
    )
}

#[cfg(test)]
mod tests {
    use super::{BRAND, Palette};
    use crate::tui::appearance::Appearance;
    use ratatui::style::Color;

    /// The TUI's brand fill is InferLab Blue; the website stylesheet owns the
    /// hex spelling (`--inferlab-blue`), so a brand refresh cannot drift the
    /// product surface silently. Per-palette accent text is exempt.
    #[test]
    fn brand_fill_matches_the_brand_stylesheet() -> Result<(), Box<dyn std::error::Error>> {
        let brand = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../website/src/styles/brand.css"
        ))?;
        let Color::Rgb(red, green, blue) = BRAND else {
            return Err("the TUI brand fill is no longer a direct RGB color".into());
        };
        let hex = format!("#{red:02X}{green:02X}{blue:02X}");
        assert!(
            brand.contains(&format!("--inferlab-blue: {hex};")),
            "the TUI brand fill {hex} drifted from the brand stylesheet"
        );
        Ok(())
    }

    #[test]
    fn a_reported_background_tints_surfaces_and_an_unknown_one_paints_none() {
        let unknown = Palette::for_appearance(Appearance::default());
        assert_eq!(unknown.panel, None);
        let light = Palette::for_appearance(Appearance {
            light: true,
            background: Some((250, 250, 248)),
        });
        assert_eq!(light.panel, Some(Color::Rgb(243, 243, 241)));
        assert_ne!(
            light.accent, unknown.accent,
            "each palette has its own accent text"
        );
    }
}

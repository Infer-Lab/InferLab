//! The terminal's appearance, read once at startup ([[ADR-0055]]): the TUI
//! asks the terminal for its colors within a bound, then falls back to
//! `COLORFGBG`, and otherwise assumes a dark background — today's palette, so
//! a terminal that does not answer loses nothing.

use std::time::Duration;

/// How long the terminal may take to report its colors. Terminals that cannot
/// answer are recognized well before it; the margin covers SSH latency.
const QUERY_BOUND: Duration = Duration::from_millis(500);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Appearance {
    pub(super) light: bool,
    /// The reported background, when the terminal answered: surfaces are then
    /// tinted from it instead of painted over an unknown color.
    pub(super) background: Option<(u8, u8, u8)>,
}

/// Query the controlling terminal. Runs before the TUI takes over the
/// terminal; the query owns raw mode for its own duration and consumes the
/// terminal's replies, so none reaches key input.
pub(super) fn detect() -> Appearance {
    let mut options = terminal_colorsaurus::QueryOptions::default();
    options.timeout = QUERY_BOUND;
    let reported = terminal_colorsaurus::color_palette(options)
        .ok()
        .map(|palette| {
            (
                palette.theme_mode() == terminal_colorsaurus::ThemeMode::Light,
                eight_bit(&palette.background),
            )
        });
    resolve(reported, std::env::var("COLORFGBG").ok().as_deref())
}

fn eight_bit(color: &terminal_colorsaurus::Color) -> (u8, u8, u8) {
    let scale = |channel: u16| u8::try_from(channel >> 8).unwrap_or(u8::MAX);
    (scale(color.r), scale(color.g), scale(color.b))
}

fn resolve(reported: Option<(bool, (u8, u8, u8))>, colorfgbg: Option<&str>) -> Appearance {
    if let Some((light, background)) = reported {
        return Appearance {
            light,
            background: Some(background),
        };
    }
    Appearance {
        light: colorfgbg.and_then(colorfgbg_is_light).unwrap_or(false),
        background: None,
    }
}

/// `COLORFGBG` is `fg;bg` (sometimes `fg;default;bg`) in the 16-color
/// palette; a background of 7 (white) or 9–15 (bright colors) is light.
fn colorfgbg_is_light(value: &str) -> Option<bool> {
    let background = value.rsplit(';').next()?.trim().parse::<u8>().ok()?;
    Some(matches!(background, 7 | 9..=15))
}

#[cfg(test)]
mod tests {
    use super::{Appearance, resolve};

    #[test]
    fn a_reported_background_decides_and_is_kept_for_tinting() {
        assert_eq!(
            resolve(Some((true, (250, 250, 248))), Some("15;0")),
            Appearance {
                light: true,
                background: Some((250, 250, 248)),
            },
            "the terminal's answer outranks COLORFGBG"
        );
    }

    #[test]
    fn without_an_answer_colorfgbg_decides_and_otherwise_dark() {
        assert!(resolve(None, Some("0;15")).light);
        assert!(resolve(None, Some("0;default;7")).light);
        assert!(!resolve(None, Some("15;0")).light);
        assert_eq!(resolve(None, Some("garbage")), Appearance::default());
        assert_eq!(resolve(None, None), Appearance::default());
        assert_eq!(resolve(None, Some("0;15")).background, None);
    }
}

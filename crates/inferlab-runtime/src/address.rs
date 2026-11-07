//! Network address spelling shared by every rendered endpoint
//! ([[RFC-0002:C-LOCAL-PLACEMENT]]).

use std::net::Ipv6Addr;

/// The `host:port` authority of an endpoint: an IPv6 address is bracketed, as
/// URLs and `host:port` options require, and every other host is written as
/// it is.
pub fn authority(host: &str, port: u16) -> String {
    if host.parse::<Ipv6Addr>().is_ok() {
        format!("[{host}]:{port}")
    } else {
        format!("{host}:{port}")
    }
}

#[cfg(test)]
mod tests {
    use super::authority;

    #[test]
    fn an_ipv6_address_is_bracketed_and_others_are_not() {
        assert_eq!(authority("2001:db8::1", 8000), "[2001:db8::1]:8000");
        assert_eq!(authority("192.0.2.10", 8000), "192.0.2.10:8000");
        assert_eq!(authority("node-a.example", 8000), "node-a.example:8000");
    }
}

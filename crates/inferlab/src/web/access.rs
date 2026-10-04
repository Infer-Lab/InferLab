//! Per-start token access ([[RFC-0012:C-ACCESS]]): the printed URL carries
//! the token once, a cookie named for this instance carries it afterward, and
//! a change must come from a page this service served.

use axum::body::Body;
use axum::extract::{Request, State};
use axum::http::{HeaderMap, HeaderValue, Method, StatusCode, header};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use std::io::Read;
use std::sync::Arc;

pub(super) struct Access {
    token: String,
    cookie: String,
}

impl Access {
    /// A 256-bit token and a per-instance cookie name, from the operating
    /// system's secure random source.
    pub(super) fn generate() -> std::io::Result<Self> {
        let mut bytes = [0_u8; 36];
        std::fs::File::open("/dev/urandom")?.read_exact(&mut bytes)?;
        let hex = |bytes: &[u8]| {
            bytes
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>()
        };
        Ok(Self {
            token: hex(&bytes[..32]),
            cookie: format!("inferlab_web_{}", hex(&bytes[32..])),
        })
    }

    pub(super) fn token(&self) -> &str {
        &self.token
    }

    fn admits(&self, presented: &str) -> bool {
        // Constant-time comparison: the token is a credential.
        presented.len() == self.token.len()
            && presented
                .bytes()
                .zip(self.token.bytes())
                .fold(0_u8, |difference, (left, right)| {
                    difference | (left ^ right)
                })
                == 0
    }
}

pub(super) async fn guard(
    State(access): State<Arc<Access>>,
    request: Request,
    next: Next,
) -> Response {
    let mut response = admit(&access, request, next).await;
    response.headers_mut().insert(
        header::REFERRER_POLICY,
        HeaderValue::from_static("no-referrer"),
    );
    response
}

async fn admit(access: &Access, request: Request, next: Next) -> Response {
    if let Some((presented, rest)) = query_token(request.uri().query()) {
        if !access.admits(&presented) {
            return unauthorized();
        }
        // Accept the URL token once, keep it in an HttpOnly cookie, and take
        // it out of the address bar.
        let location = match rest {
            Some(query) => format!("{}?{query}", request.uri().path()),
            None => request.uri().path().to_owned(),
        };
        return Response::builder()
            .status(StatusCode::SEE_OTHER)
            .header(header::LOCATION, location)
            .header(
                header::SET_COOKIE,
                format!(
                    "{}={}; HttpOnly; SameSite=Strict; Path=/",
                    access.cookie, access.token
                ),
            )
            .body(Body::empty())
            .unwrap_or_else(|_| unauthorized());
    }
    if !cookie(request.headers(), &access.cookie).is_some_and(|value| access.admits(value)) {
        return unauthorized();
    }
    if request.method() != Method::GET && !same_origin(request.headers()) {
        return (
            StatusCode::FORBIDDEN,
            "the request did not come from this console",
        )
            .into_response();
    }
    next.run(request).await
}

fn unauthorized() -> Response {
    (StatusCode::UNAUTHORIZED, "an access token is required").into_response()
}

/// The `token` query parameter and the rest of the query without it.
fn query_token(query: Option<&str>) -> Option<(String, Option<String>)> {
    let query = query?;
    let mut token = None;
    let mut rest = Vec::new();
    for pair in query.split('&') {
        match pair.strip_prefix("token=") {
            Some(value) => token = Some(value.to_owned()),
            None if !pair.is_empty() => rest.push(pair),
            None => {}
        }
    }
    token.map(|token| (token, (!rest.is_empty()).then(|| rest.join("&"))))
}

fn cookie<'a>(headers: &'a HeaderMap, name: &str) -> Option<&'a str> {
    headers
        .get_all(header::COOKIE)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(|value| value.split(';'))
        .filter_map(|pair| pair.trim().split_once('='))
        .find(|(key, _)| *key == name)
        .map(|(_, value)| value)
}

/// The Origin's host and port equal the request's Host header, which is what
/// the browser sent through any SSH forward.
fn same_origin(headers: &HeaderMap) -> bool {
    let host = headers
        .get(header::HOST)
        .and_then(|value| value.to_str().ok());
    let origin = headers
        .get(header::ORIGIN)
        .and_then(|value| value.to_str().ok())
        .and_then(|origin| origin.split_once("://"))
        .map(|(_, authority)| authority);
    matches!((host, origin), (Some(host), Some(origin)) if host == origin)
}

#[cfg(test)]
mod tests {
    use super::query_token;

    #[test]
    fn the_token_leaves_the_query_and_the_rest_stays() {
        assert_eq!(
            query_token(Some("token=abc&path=/x")),
            Some(("abc".to_owned(), Some("path=/x".to_owned())))
        );
        assert_eq!(
            query_token(Some("token=abc")),
            Some(("abc".to_owned(), None))
        );
        assert_eq!(query_token(Some("path=/x")), None);
    }
}

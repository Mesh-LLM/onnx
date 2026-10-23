//! Loopback HTTP implementation of the Mesh document_ocr_v1 provider contract.

use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
};

use anyhow::{Context, Result};
use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, Multipart, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use image::GrayImage;
use pdfium_render::prelude::*;
use serde_json::json;

use crate::ocr::{Document, Engine};

const MODEL: &str = "datalab/indra-ocr";
const MAX_FILE: usize = 64 * 1024 * 1024;
const MAX_BODY: usize = MAX_FILE + 64 * 1024;
static PDFIUM_BIND_LOCK: Mutex<()> = Mutex::new(());

#[derive(Clone)]
pub struct ServerState {
    engine: Arc<Mutex<Engine>>,
    pdfium_library: Option<PathBuf>,
}

impl ServerState {
    pub fn new(engine: Engine, pdfium_library: Option<PathBuf>) -> Self {
        Self {
            engine: Arc::new(Mutex::new(engine)),
            pdfium_library,
        }
    }
}

pub fn router(state: ServerState) -> Router {
    Router::new()
        .route("/v1/models", get(models))
        .route("/health", get(health))
        .route("/v1/documents/ocr", post(ocr))
        .layer(DefaultBodyLimit::max(MAX_BODY))
        .with_state(state)
}

async fn models() -> impl IntoResponse {
    Json(json!({"object": "list", "data": [{"id": MODEL, "object": "model"}]}))
}

async fn health() -> impl IntoResponse {
    Json(json!({"status": "ok"}))
}

struct Upload {
    file: Vec<u8>,
    model: Option<String>,
    response_format: String,
    max_pages: usize,
}

async fn read_upload(mut multipart: Multipart) -> Result<Upload, ApiError> {
    let mut upload = Upload {
        file: Vec::new(),
        model: None,
        response_format: "json".into(),
        max_pages: 0,
    };
    let mut seen_file = false;
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(ApiError::bad_request)?
    {
        match field.name() {
            Some("file") if !seen_file => {
                seen_file = true;
                upload.file = field.bytes().await.map_err(ApiError::bad_request)?.to_vec();
            }
            Some("model") if upload.model.is_none() => {
                upload.model = Some(field.text().await.map_err(ApiError::bad_request)?);
            }
            Some("response_format") => {
                upload.response_format = field.text().await.map_err(ApiError::bad_request)?;
            }
            Some("max_pages") => {
                upload.max_pages = field
                    .text()
                    .await
                    .map_err(ApiError::bad_request)?
                    .parse()
                    .map_err(ApiError::bad_request)?;
            }
            _ => {
                return Err(ApiError::bad_request(
                    "unexpected or duplicate multipart field",
                ));
            }
        }
    }
    if !seen_file || upload.file.is_empty() {
        return Err(ApiError::bad_request("file field is required"));
    }
    if upload.file.len() > MAX_FILE {
        return Err(ApiError::new(
            StatusCode::PAYLOAD_TOO_LARGE,
            "document exceeds 64 MiB",
        ));
    }
    if upload.model.as_deref().unwrap_or(MODEL) != MODEL {
        return Err(ApiError::bad_request("unsupported OCR model"));
    }
    if !matches!(
        upload.response_format.as_str(),
        "json" | "text" | "markdown"
    ) {
        return Err(ApiError::bad_request(
            "response_format must be json, text, or markdown",
        ));
    }
    if upload.max_pages > 1000 {
        return Err(ApiError::bad_request(
            "max_pages must be between 0 and 1000",
        ));
    }
    Ok(upload)
}

async fn ocr(State(state): State<ServerState>, multipart: Multipart) -> Result<Response, ApiError> {
    let upload = read_upload(multipart).await?;
    let response_format = upload.response_format.clone();
    let document = tokio::task::spawn_blocking(move || process_upload(&state, upload))
        .await
        .map_err(ApiError::internal)?
        .map_err(|error| ApiError::bad_request(format!("{error:#}")))?;
    let joined = |markdown: bool| {
        document
            .pages
            .iter()
            .map(|page| {
                if markdown {
                    page.markdown.as_str()
                } else {
                    page.text.as_str()
                }
            })
            .collect::<Vec<_>>()
            .join("\n\n")
    };
    Ok(match response_format.as_str() {
        "text" => (
            [("content-type", "text/plain; charset=utf-8")],
            joined(false),
        )
            .into_response(),
        "markdown" => (
            [("content-type", "text/markdown; charset=utf-8")],
            joined(true),
        )
            .into_response(),
        _ => Json(document).into_response(),
    })
}

fn process_upload(state: &ServerState, upload: Upload) -> Result<Document> {
    let pages = decode_pages(
        &upload.file,
        upload.max_pages,
        state.pdfium_library.as_deref(),
    )?;
    let mut engine = state
        .engine
        .lock()
        .map_err(|_| anyhow::anyhow!("OCR engine lock poisoned"))?;
    engine.recognize_document(&pages)
}

fn decode_pages(
    data: &[u8],
    max_pages: usize,
    pdfium_library: Option<&std::path::Path>,
) -> Result<Vec<GrayImage>> {
    if data.starts_with(b"%PDF-") {
        return render_pdf(data, max_pages, pdfium_library);
    }
    let image = image::load_from_memory(data)
        .context("decode document image")?
        .into_luma8();
    Ok(vec![image])
}

fn render_pdf(
    data: &[u8],
    max_pages: usize,
    library: Option<&std::path::Path>,
) -> Result<Vec<GrayImage>> {
    let _bind_lock = PDFIUM_BIND_LOCK
        .lock()
        .map_err(|_| anyhow::anyhow!("Pdfium binding lock poisoned"))?;
    let bindings = match library {
        Some(path) => Pdfium::bind_to_library(path),
        None => Pdfium::bind_to_system_library(),
    };
    let pdfium = match bindings {
        Ok(bindings) => Pdfium::new(bindings),
        Err(PdfiumError::PdfiumLibraryBindingsAlreadyInitialized) => Pdfium::default(),
        Err(error) => return Err(error).context("load Pdfium library for PDF rendering"),
    };
    drop(_bind_lock);
    let document = pdfium
        .load_pdf_from_byte_vec(data.to_vec(), None)
        .context("open uploaded PDF")?;
    let count = if max_pages == 0 {
        document.pages().len() as usize
    } else {
        max_pages.min(document.pages().len() as usize)
    };
    let config = PdfRenderConfig::new().scale_page_by_factor(200.0 / 72.0);
    document
        .pages()
        .iter()
        .take(count)
        .map(|page| Ok(page.render_with_config(&config)?.as_image()?.into_luma8()))
        .collect()
}

struct ApiError {
    status: StatusCode,
    message: String,
}

impl ApiError {
    fn new(status: StatusCode, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
        }
    }

    fn bad_request(error: impl std::fmt::Display) -> Self {
        Self::new(StatusCode::BAD_REQUEST, error.to_string())
    }

    fn internal(error: impl std::fmt::Display) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, error.to_string())
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(json!({"error": {"message": self.message}})),
        )
            .into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires a Pdfium library"]
    fn pdfium_renders_uploaded_pdf() {
        let library = std::env::var("ONNX_OCR_PDFIUM_LIBRARY").expect("ONNX_OCR_PDFIUM_LIBRARY");
        let data = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/scan.pdf"
        ))
        .expect("read PDF");
        let pages =
            decode_pages(&data, 1, Some(std::path::Path::new(&library))).expect("render PDF");
        assert_eq!(pages.len(), 1);
        assert!(pages[0].width() > 100 && pages[0].height() > 100);
    }

    #[tokio::test]
    #[ignore = "requires Indra v1.1.0 model assets and Pdfium"]
    async fn native_provider_serves_document_http_contract() {
        let model_dir = std::env::var("ONNX_OCR_MODEL_DIR").expect("ONNX_OCR_MODEL_DIR");
        let engine = Engine::load(std::path::Path::new(&model_dir)).expect("load ONNX sessions");
        let pdfium = std::fs::canonicalize(
            std::env::var("ONNX_OCR_PDFIUM_LIBRARY").expect("ONNX_OCR_PDFIUM_LIBRARY"),
        )
        .expect("Pdfium path");
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let address = listener.local_addr().expect("address");
        let task = tokio::spawn(async move {
            axum::serve(listener, router(ServerState::new(engine, Some(pdfium)))).await
        });
        for file in ["scan.png", "scan.pdf"] {
            let path = format!("{}/tests/fixtures/{file}", env!("CARGO_MANIFEST_DIR"));
            let form = reqwest::multipart::Form::new().text("model", MODEL).part(
                "file",
                reqwest::multipart::Part::bytes(std::fs::read(path).expect("sample"))
                    .file_name(file),
            );
            let response = reqwest::Client::new()
                .post(format!("http://{address}/v1/documents/ocr"))
                .multipart(form)
                .send()
                .await
                .expect("OCR response");
            let status = response.status();
            if status != StatusCode::OK {
                panic!(
                    "{file}: {status}: {}",
                    response.text().await.expect("error body")
                );
            }
            let body: serde_json::Value = response.json().await.expect("JSON document");
            assert_eq!(body["model"], MODEL, "{file}");
            assert_eq!(body["pages"][0]["page_index"], 0, "{file}");
            assert!(
                body["pages"][0]["text"]
                    .as_str()
                    .unwrap_or_default()
                    .contains("대한성공회 서울교구"),
                "{file}"
            );
            assert!(
                body["pages"][0]["markdown"]
                    .as_str()
                    .unwrap_or_default()
                    .contains("대한성공회 서울교구"),
                "{file}"
            );
        }
        task.abort();
    }
}

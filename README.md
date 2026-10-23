# ONNX plugin for Mesh LLM

This repository is the intended home for ONNX-backed inference in Mesh LLM. Its first proposed backend is [Indra OCR](https://github.com/datalab-to/indra), a CPU document parser that produces text, Markdown, and structured page layout.

## Proposed boundary

- The plugin owns ONNX Runtime sessions, pinned model assets, document preprocessing, OCR, layout, and result assembly. Rust's [`ort`](https://github.com/pykeio/ort) bindings are the first runtime to evaluate.
- Mesh owns the public document API, provider health and capability discovery, node selection, upload limits, and relay between nodes.
- A shared `document_ocr` provider contract should let other OCR plugins use the same Mesh route. A proposed first operation is `POST /v1/documents/ocr`, accepting a PDF or image and returning page-indexed Markdown, text, or JSON with boxes and reading order.
- Model weights are separate from the plugin binary. Indra's weights have a [modified OpenRAIL-M license](https://huggingface.co/datalab-to/indra-ocr/blob/main/LICENSE), so distribution and hosted-use terms must be checked before packaging or enabling them on a public mesh.

The current Mesh plugin API supports local, namespaced HTTP bindings and streamed request bodies. Mesh-wide routing of a standard document endpoint requires an explicit OCR workload/provider contract in Mesh core. The design and research notes are tracked in [DEV-26](https://linear.app/general-inference/issue/DEV-26/ocr-research).

## Status

Repository initialized for the ONNX plugin. No OCR implementation or release artifacts are available yet.

# NOTICE

## WikiRAG

WikiRAG source code is released under the MIT License. See [`LICENSE`](LICENSE).

## Tensura Wiki content

The knowledge-base content is derived from the [Tensura Wiki](https://tensura.fandom.com/).
Content derived from that wiki is licensed under the
[Creative Commons Attribution-ShareAlike 3.0 Unported license (CC BY-SA 3.0)](https://creativecommons.org/licenses/by-sa/3.0/).

When redistributing derived content, preserve the required attribution and
ShareAlike terms. This repository does not include the generated wiki dump,
vector database, chat history, or other generated personal data.

## Embedding model

WikiRAG is designed to use [gpahal/bge-m3-onnx-int8](https://huggingface.co/gpahal/bge-m3-onnx-int8),
an ONNX INT8 conversion of [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3).
Please review and comply with the license and usage terms shown in the model
card before downloading or redistributing the model files.

The model file is intentionally not included in this repository because of its
size. Users must download it separately and place it at the path documented in
the README.

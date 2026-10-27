# Multimodal Embedding with EmbeddingGemma 2 and OpenVINO

This tutorial shows how to run the full multimodal embedding pipeline of [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) with OpenVINO.

EmbeddingGemma 2 is an open multimodal embedding model from Google DeepMind. It maps text, code, images, video, and audio into a single 768-dimensional vector space. The full model has 740M parameters: a 270M text backbone plus modular vision and audio encoders.

## Notebook Contents

The notebook demonstrates the full multimodal path and validates every OpenVINO embedding against the original Sentence Transformers model:

1. Install OpenVINO, OpenVINO Tokenizers, Transformers, PyTorch, and Sentence Transformers.
2. Create small synthetic image, audio, and video samples.
3. Inspect the processor features for text, image, video, audio, and interleaved inputs.
4. Export the model parts to OpenVINO IR: text embedder, image vision tower, video vision tower, vision embedder, audio tower, audio embedder, and language model.
5. Select an OpenVINO device and compile the exported IR parts. `NPU` is listed when the OpenVINO NPU plugin and driver are installed; if NPU compilation fails on your driver, select `CPU` or `GPU`.
6. Merge hard text embeddings with image, video, and audio soft tokens in Python.
7. Run OpenVINO multimodal inference and compare text, image, video, audio, and interleaved embeddings with Sentence Transformers.
8. Apply Matryoshka dimension truncation.
9. Benchmark the OpenVINO IR pipeline on the selected device against the Sentence Transformers model on CPU.

The notebook also documents the design logic: the exported OpenVINO IR parts are composed in Python in the same order as `EmbeddingGemma2Model.forward` in Transformers, and the scope and limitations of that composition are stated explicitly.

The current OpenVINO converter cannot export the complete multimodal model as one IR, so this notebook exports the parts separately and composes them at inference time.

The committed CPU benchmark compares the OpenVINO IR pipeline with the PyTorch Sentence Transformers reference. If `NPU` is selected, the same benchmark compares NPU inference with the non-NPU PyTorch CPU reference.

A measured NPU run on an Intel Core Ultra 7 155H with NPU driver `32.0.100.5540` and OpenVINO `2026.4.1` is recorded in the notebook. For batch size `1` synthetic samples, NPU audio inference was `1.18x` faster than the OpenVINO CPU pipeline with cosine `0.992069`. Text latency depends on sequence length: a 14-token query is slower on NPU (`0.17x`) because the NPU graph is static at `715` tokens, while a 715-token input is `6.02x` faster on NPU with cosine `0.999933`. The image and video vision towers are not part of the measured NPU table because the tested NPU driver returned an incorrect output shape for the exported vision tower.

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/embeddinggemma-2/README.md" />

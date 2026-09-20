# Edge-Optimized Diagnostic Imaging Classifier

Benchmarking accuracy vs. efficiency trade-offs when compressing a pretrained
image classifier for deployment on low-power, low-resource hardware — with
malaria blood cell classification as the initial proof case.

## Motivation

Rapid, point-of-care diagnostics remain out of reach in many low-resource
settings not because the underlying detection methods don't exist, but
because deploying them requires hardware most clinics can't afford or power
reliably. This project asks a narrow, concrete question: **how much
accuracy do you actually lose when you compress a diagnostic imaging model
down to run on cheap, low-power hardware — and is that trade-off worth
it for real-world deployment?**

This is built as a first step toward a broader goal: point-of-care,
imaging-based diagnostic instrumentation for low-resource African health
systems, starting with proof-of-concept work on public datasets before
scaling to real clinical phase-imaging data.

## What this does

1. Loads a public malaria cell image dataset (infected vs. uninfected red
   blood cells)
2. Fine-tunes a pretrained MobileNetV2 classifier on the task
3. Benchmarks the model: accuracy, size, inference speed (**baseline**)
4. Applies edge optimization (post-training static quantization, with
   per-channel calibration)
5. Benchmarks again (**compressed**)
6. Reports the accuracy/size/speed trade-off directly
7. Exports the final model to ONNX for actual edge deployment

## Results

| Metric | Baseline (fp32) | Compressed (int8) | Change |
|---|---|---|---|
| Accuracy | 97.29% | 92.71% | -4.58 pts |
| Model size | 8.47 MB | 2.53 MB | 70.2% smaller |
| Inference time | 308.52 ms/batch | 389.63 ms/batch | 26% slower |

## Key findings

**Naive per-tensor quantization broke the model outright** (accuracy collapsed
to 57.4%, barely above random for a 2-class task) before this fix was applied.
The cause: MobileNetV2 is built almost entirely from depthwise separable
convolutions, and per-tensor quantization forces one shared scale across an
entire layer's weights — depthwise conv layers have highly variable
per-channel weight distributions, so a single shared scale destroys a lot of
information. Switching to **per-channel quantization** (`per_channel=True`),
which gives each channel its own scale, brought accuracy back to within 4.6
points of the original model while keeping most of the size reduction. This
is a documented, architecture-specific weakness of MobileNet-style models
under naive quantization, not a one-off bug.

**Compression didn't translate to faster inference on this hardware** —
in fact it got slightly slower (308ms → 390ms/batch). This is a real and
fairly common finding, not a broken result: int8 speedups depend on the
runtime having optimized low-precision kernels for the specific CPU
instruction set in use, and on this environment those gains didn't
materialize. The takeaway is important for real deployment decisions: **size
reduction and speed gains are separate claims that need to be verified
independently** — a smaller model is not automatically a faster one, and
production point-of-care hardware would need to be benchmarked directly
rather than assuming compression implies speed.

## Debugging journey

This project involved more troubleshooting than the final pipeline suggests
— documented here since the fixes are as instructive as the results:

1. **Dataset had a duplicate nested folder** — the Kaggle malaria dataset
   contains `cell_images/cell_images/`, which `ImageFolder` picked up as a
   spurious third class, causing a silent label mismatch that crashed
   training with a cryptic CUDA assert error. Fixed by filtering samples to
   only the two real class folders.
2. **Dynamic quantization barely changed model size** — PyTorch's dynamic
   quantization only compresses `nn.Linear` layers by default, and
   MobileNetV2 is almost entirely `Conv2d`. Switched to **static
   quantization**, the correct method for CNN-based vision models.
3. **`onnxscript` / `protobuf` version conflicts** — the newer PyTorch ONNX
   export path depends on packages not preinstalled in Colab, and the
   `protobuf` version pulled in by one package conflicted with another's
   requirement. Resolved by explicitly upgrading both together in one
   install command.
4. **A broken ONNX export produced a 0.24 MB file** (should be ~8-9 MB) —
   traced to PyTorch's newer default "dynamo" export path silently failing
   on this architecture. Fixed by forcing the legacy exporter (`dynamo=False`).
5. **Quantized model accuracy collapsed to ~57%** (near random for a 2-class
   task) — caused by per-tensor quantization on MobileNetV2's depthwise
   separable convolutions, which have highly variable per-channel weight
   distributions that one shared scale can't represent well. Fixed with
   `per_channel=True`.

## Why this matters for low-resource deployment

A model that's slightly less accurate but dramatically smaller and faster
can be the difference between "runs only on a lab workstation" and "runs
on a $50 single-board computer with no internet connection." That gap is
the actual bottleneck for point-of-care diagnostics in many settings —
not whether accurate models exist, but whether they can physically run
where they're needed.

## Tech stack

- PyTorch / torchvision — model training and quantization
- ONNX / ONNX Runtime — edge-deployable model export
- Google Colab — development environment (GPU-accelerated fine-tuning)

## Dataset

[NIH Malaria Cell Images Dataset](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria)
(~27,000 labeled thin blood smear cell images, via Kaggle)

Future work will apply this same pipeline to real quantitative phase
imaging (QPI) datasets from published research (see Roadmap below).

## How to run

1. Open `edge_optimization_starter.py` in Google Colab
2. Copy each `# ==== SECTION ====` block into its own cell, in order
3. Run top to bottom — the dataset downloads automatically via `kagglehub`

Or locally:
```bash
pip install -r requirements.txt
python edge_optimization_starter.py
```

## Roadmap

- [x] Baseline pipeline on public malaria dataset
- [x] Static quantization benchmarking (per-channel)
- [ ] Apply pipeline to real quantitative phase imaging (QPI) data
      (pending data access from published research groups)
- [ ] Explore additional compression methods (pruning, structured sparsity)
- [ ] Benchmark on actual low-power hardware (e.g. Raspberry Pi)
- [ ] Extend to sickle cell disease classification

## Background

This project builds on earlier work in quantitative phase imaging for
point-of-care red blood cell diagnostics (QPI-net), reapplying the same
imaging + machine learning skill set toward antimicrobial resistance and
rapid diagnostic instrumentation more broadly.

## Author

Opeyemi Olawuwo
Biomedical Technology, Federal University of Technology, Akure (FUTA)

## License

MIT — see `LICENSE`

# Hardware and software stack

## Hardware
- GPU: NVIDIA A100-SXM4-80GB (Google Colab Pro)
- CPU: Colab default
- RAM: 80 GB high-RAM Colab Pro instance

## Software
| Component       | Version                  |
| --------------- | ------------------------ |
| CUDA            | 13.0                     |
| PyTorch         | 2.11.0+cu128             |
| vLLM            | 0.11.2                   |
| AWQ quantizer   | AWQ-Marlin 4-bit         |
| transformers    | 4.x compatible           |
| Model           | Qwen/Qwen2.5-72B-Instruct-AWQ |

## Measured runtime

| Dataset | Questions | Wall-clock         | GPU-hours | Throughput     | Mean latency  |
| ------- | --------- | ------------------ | --------- | -------------- | ------------- |
| WebQSP  | 1,639     | 9,112.05 s (2.53 h)| 2.53      | 10.79 q / min  | 5.56 s / q    |
| CWQ     | 3,531     | 16,058.75 s (4.46 h)| 4.46     | 13.19 q / min  | 4.55 s / q    |

Peak GPU memory at model load (reported by vLLM): **38.76 GiB**.

## Reproducibility caveats specific to the hardware
- Google Colab schedules A100 instances on shared-tenant infrastructure;
  precise latency/throughput numbers will vary by run. The values above are
  from a single timed run reported in the paper.
- The `vllm` version pinned in `requirements.txt` is the one used for the
  published numbers. Different versions can change tokenization and decoding
  behavior subtly.

# DeepSeek-V2-Lite Architecture Inputs

These files were copied without modification on2026-10-28 from the model
definition used by the recorded experiments:
`/data/haiqwa/zevin_nfs/code/verl/examples/resources/DeepSeek-V2-Lite`.
The upstream README and LICENSE are retained alongside the definitions.
No pretrained weights or tokenizer payloads are included. The checkpoint
grouping experiments construct random-initialized models using synthetic tokens.

| File | SHA-256 |
| --- | --- |
| config.json | f346286b0f1c8b044252fd54cb4fa78b9fab6472a6e8bebb9edfe03d414ea03d |
| configuration_deepseek.py | 6de5e7445ab490a16482027b131d5bd6a003da9b58a182a11432115fac84ce59 |
| modeling_deepseek.py | 7d8e5221095286eea991137760893fd7ba52727c0b4ebf48ec09e8bc56b45b9c |

`CKPT_DEEPSEEK_MODEL_DIR` can select another compatible model definition.
Pretrained-weight experiments require separately obtained weights and are
outside this retained random-initialization artifact.

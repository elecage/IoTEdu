"""Utility to regenerate the C source that stores the neural network weights.

Edit the NumPy arrays in this script (or replace them with values exported
from your training pipeline) and run the script. It overwrites
``stm32_ai_example/Core/Src/ai_model.c`` with updated weights while keeping the
inference logic identical.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "Core" / "Src" / "ai_model.c"

DENSE1_WEIGHTS = np.array(
    [
        [0.42, -0.37, 0.58, -0.21],
        [-0.12, 0.33, 0.44, 0.18],
        [0.31, 0.09, -0.47, 0.52],
        [0.27, -0.29, 0.18, 0.61],
        [-0.55, 0.12, 0.34, -0.19],
        [0.16, -0.48, 0.23, 0.37],
        [0.08, 0.57, -0.21, 0.42],
        [-0.24, 0.15, 0.19, -0.31],
    ],
    dtype=np.float32,
)

DENSE1_BIAS = np.array([0.12, -0.05, 0.03, 0.08, -0.11, 0.02, 0.06, -0.04], dtype=np.float32)

DENSE2_WEIGHTS = np.array(
    [
        [0.44, -0.36, 0.51, -0.27, 0.32, -0.18, 0.49, -0.12],
        [-0.22, 0.41, -0.39, 0.33, -0.28, 0.26, -0.31, 0.37],
        [0.17, -0.22, 0.28, -0.19, 0.41, -0.29, 0.24, -0.16],
    ],
    dtype=np.float32,
)

DENSE2_BIAS = np.array([0.05, -0.03, 0.02], dtype=np.float32)

HEADER = """#include \"ai_model.h\"\n\n#include <math.h>\n\n"""

FOOTER = """
static float relu(float x)
{
    return (x > 0.0f) ? x : 0.0f;
}

static void softmax(const float *input, float *output, int length)
{
    float max_value = input[0];
    for (int i = 1; i < length; ++i)
    {
        if (input[i] > max_value)
        {
            max_value = input[i];
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < length; ++i)
    {
        output[i] = expf(input[i] - max_value);
        sum += output[i];
    }

    const float reciprocal = 1.0f / sum;
    for (int i = 0; i < length; ++i)
    {
        output[i] *= reciprocal;
    }
}

void ai_model_forward(const float *input, float *output)
{
    float hidden[AI_MODEL_HIDDEN_DIM];

    for (int i = 0; i < AI_MODEL_HIDDEN_DIM; ++i)
    {
        float acc = dense1_bias[i];
        for (int j = 0; j < AI_MODEL_INPUT_DIM; ++j)
        {
            acc += dense1_weights[i][j] * input[j];
        }
        hidden[i] = relu(acc);
    }

    float logits[AI_MODEL_OUTPUT_DIM];
    for (int i = 0; i < AI_MODEL_OUTPUT_DIM; ++i)
    {
        float acc = dense2_bias[i];
        for (int j = 0; j < AI_MODEL_HIDDEN_DIM; ++j)
        {
            acc += dense2_weights[i][j] * hidden[j];
        }
        logits[i] = acc;
    }

    softmax(logits, output, AI_MODEL_OUTPUT_DIM);
}
"""


def format_matrix(name: str, matrix: np.ndarray) -> str:
    rows = []
    for row in matrix:
        values = ", ".join(f"{value:.6f}f" for value in row)
        rows.append(f"    {{{values}}}")
    body = ",\n".join(rows)
    return (
        f"static const float {name}[{matrix.shape[0]}][{matrix.shape[1]}] = {{\n"
        f"{body}\n" "};\n\n"
    )


def format_vector(name: str, vector: np.ndarray) -> str:
    values = ", ".join(f"{value:.6f}f" for value in vector)
    return f"static const float {name}[{vector.shape[0]}] = {{{values}}};\n\n"


def main() -> None:
    content = HEADER
    content += format_matrix("dense1_weights", DENSE1_WEIGHTS)
    content += format_vector("dense1_bias", DENSE1_BIAS)
    content += format_matrix("dense2_weights", DENSE2_WEIGHTS)
    content += format_vector("dense2_bias", DENSE2_BIAS)
    content += FOOTER

    TARGET.write_text(content)
    print(f"Updated {TARGET.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

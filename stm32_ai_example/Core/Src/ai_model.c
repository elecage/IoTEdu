#include "ai_model.h"

#include <math.h>

static const float dense1_weights[AI_MODEL_HIDDEN_DIM][AI_MODEL_INPUT_DIM] = {
    {0.42f, -0.37f, 0.58f, -0.21f},
    {-0.12f, 0.33f, 0.44f, 0.18f},
    {0.31f, 0.09f, -0.47f, 0.52f},
    {0.27f, -0.29f, 0.18f, 0.61f},
    {-0.55f, 0.12f, 0.34f, -0.19f},
    {0.16f, -0.48f, 0.23f, 0.37f},
    {0.08f, 0.57f, -0.21f, 0.42f},
    {-0.24f, 0.15f, 0.19f, -0.31f},
};

static const float dense1_bias[AI_MODEL_HIDDEN_DIM] = {
    0.12f, -0.05f, 0.03f, 0.08f, -0.11f, 0.02f, 0.06f, -0.04f,
};

static const float dense2_weights[AI_MODEL_OUTPUT_DIM][AI_MODEL_HIDDEN_DIM] = {
    {0.44f, -0.36f, 0.51f, -0.27f, 0.32f, -0.18f, 0.49f, -0.12f},
    {-0.22f, 0.41f, -0.39f, 0.33f, -0.28f, 0.26f, -0.31f, 0.37f},
    {0.17f, -0.22f, 0.28f, -0.19f, 0.41f, -0.29f, 0.24f, -0.16f},
};

static const float dense2_bias[AI_MODEL_OUTPUT_DIM] = {
    0.05f, -0.03f, 0.02f,
};

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

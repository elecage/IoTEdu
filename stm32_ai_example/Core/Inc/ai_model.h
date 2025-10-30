#ifndef AI_MODEL_H
#define AI_MODEL_H

#ifdef __cplusplus
extern "C" {
#endif

#define AI_MODEL_INPUT_DIM 4
#define AI_MODEL_HIDDEN_DIM 8
#define AI_MODEL_OUTPUT_DIM 3

void ai_model_forward(const float *input, float *output);

#ifdef __cplusplus
}
#endif

#endif /* AI_MODEL_H */

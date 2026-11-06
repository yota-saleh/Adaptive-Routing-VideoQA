# Dataset access

- MSRVTT-QA question-answer annotations: https://github.com/xudejing/video-question-answering
- MSR-VTT videos: https://huggingface.co/datasets/shiyili1111/MSR-VTT

Before training, questions were lowercased and whitespace-normalised, and repeated questions about the same video were removed (keeping the first occurrence). The same filter also dropped the questions of training video 0. The resulting splits contain 153,310 training, 11,794 validation, and 70,181 test questions.

Raw videos, annotations, extracted features, and answer vocabularies are not redistributed here.

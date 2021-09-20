# download the model for language detection
you can download the finetuned model for language detection [here](https://github.com/jechoi2021/13_languages_detection_XLM-R/releases/tag/13_languages_detection_XLM-R_v1.0)

# About the fine-tuned model

This model is fine-tuned for 13 language in total using [XLM-RoBERTa](https://github.com/huggingface/transformers/blob/master/docs/source/model_doc/xlmroberta.rst) which is a model for cross-lingual (100 languages) representation i.e. it is very likely to use it for other languages as well after short fine-tuning process.

For classification purpose [XLMRobertaForSequenceClassfication](https://huggingface.co/transformers/model_doc/xlmroberta.html#xlmrobertaforsequenceclassification) is used in order to train a classifier.

[XL-WiC dataset](https://aclanthology.org/2020.emnlp-main.584/) is used for fine-tuning, while it has been pre-processed in order to get a pair (language, text) only.

For more details, please refer to [here](https://github.com/jechoi2021/13_languages_detection_XLM-R/releases/tag/13_languages_detection_XLM-R_v1.0)

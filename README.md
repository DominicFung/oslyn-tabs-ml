# oslyn-tabs-ml

Oslyn Tabs is a NextJS and React Native (iOS & Android) app that takes externally available chord sheets / tabs and turns them into a slide show.
This is done to *maximize* screen and keep the band in sync (only 2-3 lines are shown at a time). However, today, one person needs to "turn" the page manually. This is very painful for any solo musician.

**Use case**: Oslyn Tabs needs to turn the page at the appropriate time, automating the need to take hands off the guitar or musical instrument.

## Approach 1: ASR

Leverage existing (pre-trained) speech-to-text models [via huggingface](https://huggingface.co/models?pipeline_tag=automatic-speech-recognition&sort=trending) ie:

- [x] Whisper by OpenAI
- [ ] Seamless m4t by Facebook
- [ ] Paraformer by Funasr
- [ ] Wav2vec2 by Facebook

Experimentation of different pre-trained audio speech recognition (ASR) models.

### Requirements

1. Open Source - No restrictions
2. Ensure that chosen model's can be **fine-tuned** ie, provide more musically oriented data
3. Ensure they can be fine-tuned (additional training) on Sagemaker
4. Ensure they can be deployed to device (web + iOS + Android) and NOT server-side. Maybe via ONNX convert to tflite?

### 1. Use TFLite

Since Whisper and most advanced ASR are built with PyTorch, we will turn the model (.pt) into ONNX (.onnx). Then turn onnx into Tensorflow (.pb), then TF into TFLite. A lot of react-native and Web libraries can already handle TFLite. This was demo'd back in the early days of Oslyn. Cons of this approach: PyTorch is generally a lot more advanced (using higher opsets (14-18), while Tensorflow has stopped implementing these higher level opsets.)

- [x] Run OpenAI/Whisper in Python
- [x] Convert to ONNX
- [ ] ONNX -> Tensorflow (opset issue - v12 vs v14+)
- [ ] Tensorflow -> TFLite
- [ ] Web integration test
- [ ] React Native Integration test

For future study:

- React Native: https://github.com/mrousavy/react-native-fast-tflite
- Web: https://www.npmjs.com/package/@tensorflow/tfjs-tflite

### 2. Use Onnx Runtime

Once compiled to ONNX, we can check integration using onnxruntime. There seems to be a runtime for both web and react-native.

- Examples: https://github.com/microsoft/onnxruntime-inference-examples
- https://github.com/microsoft/onnxruntime-inference-examples/blob/main/js/ort-whisper/README.md


- [x] Run OpenAI/Whisper in Python
- [x] Convert to ONNX
- [x] Web integration test
- [ ] React Native Integration test
"use client"

import * as ort from 'onnxruntime-web'
import { useEffect, useRef, useState } from 'react'

class Whisper {
  sess: ort.InferenceSession | null
  min_length: Int32Array
  max_length: Int32Array
  num_return_sequences: Int32Array
  length_penalty: Float32Array
  repetition_penalty: Float32Array
  attention_mask: Int32Array

  constructor(url: string, cb: any) {
      ort.env.logLevel = "error";
      this.sess = null;

      // semi constants that we initialize once and pass to every run() call
      this.min_length = Int32Array.from({ length: 1 }, () => 1);
      this.max_length = Int32Array.from({ length: 1 }, () => 448);
      this.num_return_sequences = Int32Array.from({ length: 1 }, () => 1);
      this.length_penalty = Float32Array.from({ length: 1 }, () => 1.);
      this.repetition_penalty = Float32Array.from({ length: 1 }, () => 1.);
      this.attention_mask = Int32Array.from({ length: 1 * 80 * 3000 }, () => 0);

      const opt = {
          executionProviders: ["wasm"],
          logSeverityLevel: 0,
          logVerbosityLevel: 0
      } as ort.InferenceSession.SessionOptions
      ort.InferenceSession.create(url, opt).then((s) => {
          this.sess = s;
          cb();
      }, (e) => { cb(e); })
  }

  async run(audio_pcm: ort.TypedTensor<"float32">, beams = 1) {
      if (!this.sess) throw "no session available"
      // clone semi constants into feed. The clone is needed if we run with ort.env.wasm.proxy=true
      const feed = {
          "audio_pcm": audio_pcm,
          "max_length": new ort.Tensor(new Int32Array(this.max_length), [1]),
          "min_length": new ort.Tensor(new Int32Array(this.min_length), [1]),
          "num_beams": new ort.Tensor(Int32Array.from({ length: 1 }, () => beams), [1]),
          "num_return_sequences": new ort.Tensor(new Int32Array(this.num_return_sequences), [1]),
          "length_penalty": new ort.Tensor(new Float32Array(this.length_penalty), [1]),
          "repetition_penalty": new ort.Tensor(new Float32Array(this.repetition_penalty), [1]),
          "attention_mask": new ort.Tensor(new Int32Array(this.attention_mask), [1, 80, 3000]),
      }
      console.log(this.sess)
      console.log(feed)

      console.log("transcribe started")
      const result = await this.sess.run(feed)
      console.log("transcribe ended")

      console.log(result)
      return result
  }
}

export default function Onnx() {

  function log(i: string) { 
    let d = document.getElementById('status')?.innerText 
    if (d) d += `\n[${performance.now().toFixed(2)}] ` + i; 
  }

  const kSampleRate = 16000;
  const kIntervalAudio_ms = 1000;
  const kSteps = kSampleRate * 30;
  const kDelay = 100;
  const kModel = "models/onnx/v3/whisper_cpu_int8_cpu-cpu_model.onnx" //"whisper_cpu_int8_0_model.onnx";

  // ort session
  let sess: Whisper;

  // audio context
  var context: AudioContext|null = null;
  let mediaRecorder: MediaRecorder|undefined

  // stats
  let total_processing_time = 0;
  let total_processing_count = 0;

  // some dom shortcuts
  let record: any;
  let transcribe: any;
  let progress: any;
  let audio_src: any;

  // transcribe active
  function busy() {
      transcribe.disabled = true;
      progress.parentNode.style.display = "block";
      (document.getElementById("outputText") as HTMLTextAreaElement).value = "";
      document.getElementById('latency')!.innerText = "";
  }

  // transcribe done
  function ready() {
      transcribe.disabled = false;
      progress.style.width = "0%";
      progress.parentNode.style.display = "none";
  }

  useEffect(() => {
    ort.env.wasm.wasmPaths = {
      'ort-wasm.wasm': '/models/onnx/dist/ort-wasm.wasm',
      'ort-wasm-simd.wasm': '/models/onnx/dist/ort-wasm-simd.wasm',
      'ort-wasm-threaded.wasm': '/models/onnx/dist/ort-wasm-threaded.wasm',
    }

    // called when document is loaded
    audio_src = document.querySelector('audio');
    record = document.getElementById('record');
    transcribe = document.getElementById('transcribe');
    progress = document.getElementById('progress');
    transcribe.disabled = true;
    progress.parentNode.style.display = "none";

    console.log("document elements loaded.")

    // audio_src.src = "colorado_river_crisis_is_hitting_home.mp3";

    // click on Record
    record.addEventListener("click", (e: any) => {
        if (e.currentTarget.innerText == "Record") {
            e.currentTarget.innerText = "Stop Recording";
            startRecord();
        }
        else {
            e.currentTarget.innerText = "Record";
            stopRecord();
        }
    });

    // click on Transcribe
    transcribe.addEventListener("click", () => {
        console.log("transcribe clicked")
        transcribe_file();
    });

    // drop file
    document.getElementById("file-upload")!.onchange = function (evt) {
        let target = evt.target || (window.event as any).src, files = target.files;
        audio_src.src = URL.createObjectURL(files[0]);
    }

    log("loading model");
    try {
        sess = new Whisper(kModel, (e: any) => {
            if (e === undefined) {
                log(`${kModel} loaded, ${ort.env.wasm.numThreads} threads`);
                ready();
            } else {
                log(`Error: ${e}`);
            }
        });

        context = new AudioContext({
            sampleRate: kSampleRate,
            channelCount: 1,
            echoCancellation: false,
            autoGainControl: true,
            noiseSuppression: true,
        } as AudioContextOptions);
        if (!context) {
            throw new Error("no AudioContext, make sure domain has access to Microphone");
        }
    } catch (e) {
        log(`Error: ${e}`);
    }
  }, [])

  // report progress
  function update_status(t: number) {
    total_processing_time += t;
    total_processing_count += 1;
    const avg = 1000 * 30 * total_processing_count / total_processing_time;
    document.getElementById('latency')!.innerText = `${avg.toFixed(1)} x realtime`;
  }

  function sleep(ms: number) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  // process audio buffer
  async function process_audio(audio: Float32Array, starttime: number, idx: number, pos: number) {
    if (idx < audio.length) {
        // not done
        try {
            // update progress bar
            progress.style.width = (idx * 100 / audio.length).toFixed(1) + "%";
            progress.textContent = progress.style.width;
            await sleep(kDelay);

            // run inference for 30 sec
            const xa = audio.slice(idx, idx + kSteps);
            const start = performance.now();
            const ret = (await sess.run(new ort.Tensor(xa, [1, xa.length])))!
            console.log(ret)
            const diff = performance.now() - start;
            update_status(diff);

            // append results to textarea 
            const textarea = document.getElementById('outputText') as HTMLTextAreaElement;
            textarea.value += `${ret.str.data[0]}\n`;
            textarea.scrollTop = textarea.scrollHeight;
            await sleep(kDelay);
            process_audio(audio, starttime, idx + kSteps, pos + 30);
        } catch (e) {
            log(`Error: ${e}`);
            ready();
        }
    } else {
        // done with audio buffer
        const processing_time = ((performance.now() - starttime) / 1000);
        const total = (audio.length / kSampleRate);
        log(`${document.getElementById('latency')!.innerText}, total ${processing_time.toFixed(1)}sec for ${total.toFixed(1)}sec`);
        ready();
    }
  }

  // transcribe audio source
  async function transcribe_file() {
    if (audio_src.src == "") {
        log("Error: set some Audio input");
        return;
    }

    busy();
    log("start transcribe ...");
    try {
        const buffer = await (await fetch(audio_src.src)).arrayBuffer();
        const audioBuffer = await context!.decodeAudioData(buffer);
        var offlineContext = new OfflineAudioContext(audioBuffer.numberOfChannels, audioBuffer.length, audioBuffer.sampleRate);
        var source = offlineContext.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(offlineContext.destination);
        source.start();
        const renderedBuffer = await offlineContext.startRendering();
        const audio = renderedBuffer.getChannelData(0);
        process_audio(audio, performance.now(), 0, 0);
    }
    catch (e) {
        log(`Error: ${e}`);
        ready();
    }
  }

  // start recording
  async function startRecord() {
    if (mediaRecorder === undefined) {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
            mediaRecorder = new MediaRecorder(stream);
        } catch (e) {
            record.innerText = "Record";
            log(`Access to Microphone, ${e}`);
        }
    }
    let recording_start = performance.now();
    let chunks: BlobPart[] = [];

    mediaRecorder!.ondataavailable = (e) => {
        chunks.push(e.data);
        document.getElementById('latency')!.innerText = `recorded: ${((performance.now() - recording_start) / 1000).toFixed(1)}sec`;
    }

    mediaRecorder!.onstop = () => {
        const blob = new Blob(chunks, { 'type': 'audio/ogg; codecs=opus' });
        log(`recorded ${((performance.now() - recording_start) / 1000).toFixed(1)}sec audio`);
        audio_src.src = window.URL.createObjectURL(blob);
    };
    mediaRecorder!.start(kIntervalAudio_ms);
  }

  // stop recording
  function stopRecord() {
    if (mediaRecorder) {
        mediaRecorder.stop();
        mediaRecorder = undefined;
    }
  }

  return <>
    <div className="container">
      <h2>onnxruntime-web whisper demo</h2>
      <br />
      <div className="d-grid gap-3">
          <div className="row">
              <div className="col-md-4 rounded-block">
                  <h4>Audio Input</h4>
                  <form>
                      <div className="form-group ">
                          <input type="file" id="file-upload" className="form-control-file" />
                      </div>
                      <br />
                      <p>OR</p>
                      <button id="record" className="btn btn-primary" type="button">Record</button>
                  </form>
              </div>

              <div className="col-md-4 rounded-block">
                  <h4>Transcribe</h4>
                  <div id="record_play">
                      <audio id="record_play_audio" controls></audio>
                  </div>
                  <br />
                  <button id="transcribe" className="btn btn-primary" type="button">Transcribe</button>
              </div>

              <div className="col-md-4 rounded-block">
                  <h4>Stats</h4>
                  <div className="bg-light">
                      <div id="latency"></div>
                  </div>
              </div>
          </div>
          <div className="row gap-3">
              <div className="progress">
                  <div id="progress" className="progress-bar progress-bar-striped" role="progressbar" 
                  style={{width: "0%"}}></div>
              </div>
          </div>
          <div className="row gap-3">
              <textarea id="outputText" className="form-control" rows={12} readOnly></textarea>
          </div>
          <div className="row gap-3">
              <div id="status" className='text-white' style={{font: "1em consolas"}}></div>
          </div>
      </div>
    </div>
  </>
}
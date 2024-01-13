"use client"

import * as ort from 'onnxruntime-web'
import { useEffect, useRef, useState } from 'react'

const kSampleRate = 16000;
const kIntervalAudio_ms = 1000;
const kSteps = kSampleRate * 30;
const kDelay = 100;
const kModel = "models/onnx/v3/whisper_cpu_int8_cpu-cpu_model.onnx"

export default function Onnx() {

  //const [inferenceSession, setInferenceSession] = useState<ort.InferenceSession|null>(null)
  const [ sess, setSess ] = useState<Whisper>()

  const main = async () => {
    ort.env.wasm.wasmPaths = {
      'ort-wasm.wasm': '/models/onnx/dist/ort-wasm.wasm',
      'ort-wasm-simd.wasm': '/models/onnx/dist/ort-wasm-simd.wasm',
      'ort-wasm-threaded.wasm': '/models/onnx/dist/ort-wasm-threaded.wasm',
    }

    log("loading model")
    try {
      setSess(new Whisper(kModel, (e: any) => {
        if (e === undefined) {
          log(`${kModel} loaded, ${ort.env.wasm.numThreads} threads`)
          ready()
        } else { log(`Error: ${e}`) }
      }))

      setContext(new AudioContext({
        sampleRate: kSampleRate,
        channelCount: 1,
        echoCancellation: false,
        autoGainControl: true,
        noiseSuppression: true,
      } as AudioContextOptions))

      if (!context) {
        throw new Error("no AudioContext, make sure domain has access to Microphone");
      }
    } catch (e) {
      log(`Error: ${e}`);
    }
  }

  useEffect(() => {  main(); getMicrophone(); return () => { closeMicrophone() } }, [])

  const [context, setContext] = useState<AudioContext|null>(null)
  const [mediaRecorder, setMediaRecorder] = useState<MediaRecorder|null>(null)
  const [ isLoading, setIsLoading ] = useState(false)
  const [isRecording, setIsRecording] = useState(false)
  const [timerStart, setTimerStart] = useState<number>(0)
  const [logText, setLogText] = useState("")

  // helpers
  const log = (i: string) => { setLogText((p) => { return `${p}\n[${performance.now().toFixed(2)}] ${i}` }) }
  const sleep = (ms: number) => { return new Promise(resolve => setTimeout(resolve, ms)) }
  const busy = () => { setIsLoading(true) }
  const ready = () => { setIsLoading(false) }
  const update_status = (d: number) => {  }

  const getMicrophone = async () => {
    if (mediaRecorder === null) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
        setMediaRecorder(new MediaRecorder(stream))
      } catch (e) {
        setIsRecording(false)
        log(`Access to Microphone, ${e}`);
      }
    }
  }

  const closeMicrophone = () => {
    mediaRecorder && mediaRecorder.stop()
    mediaRecorder? mediaRecorder.ondataavailable = null : console.log("could not deregister recorder.ondataavailable")
    mediaRecorder? mediaRecorder.onstop = null : console.log("could not deregister recorder.onstop")
    setMediaRecorder(null)
    console.log(`closed Microphone!`)
  }

  const startRecord = async () => {
    if (mediaRecorder === null) { log("ERROR no media recorder found."); return }
    if (!audioRef || !audioRef.current) { log("ERROR audio reference not found."); return }
    setIsRecording(true)
      
    let recording_start = performance.now();
    let chunks: BlobPart[] = [];

    mediaRecorder.ondataavailable = (e) => { chunks.push(e.data) }

    mediaRecorder.onstop = () => {
      const blob = new Blob(chunks, { 'type': 'audio/ogg; codecs=opus' });
      log(`recorded ${((performance.now() - recording_start) / 1000).toFixed(1)}sec audio`);
      audioRef.current!.src = window.URL.createObjectURL(blob);
      setIsRecording(false)
    }
    mediaRecorder.start(kIntervalAudio_ms);
  }

  const stopRecord = () => {
    if (mediaRecorder) {
      setIsRecording(false)
      mediaRecorder.stop()
      setMediaRecorder(null)
      let time = performance.now() - timerStart
      log(`Recording stopped, ${time}`)
    }
  }

  const audioRef = useRef<HTMLAudioElement | null>(null);

  // transcribe audio source
  async function transcribe_file() {
    if (!audioRef || !audioRef.current || audioRef.current.src == "") {
      log("Error: set some Audio input"); return;
    }

    if (!context) { log("Error: context is null"); return}

    busy();
    log("start transcribe ...");
    try {
        const buffer = await (await fetch(audioRef.current.src)).arrayBuffer();
        const audioBuffer = await context.decodeAudioData(buffer);
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

  // process audio buffer
  async function process_audio(audio: Float32Array, starttime: number, idx: number, pos: number) {
    if (!sess) { log("Error: sess is empty"); return }
    if (idx < audio.length) {
      // not done
      try {
        await sleep(kDelay);

        // run inference for 30 sec
        const xa = audio.slice(idx, idx + kSteps);
        const start = performance.now();
        const ret = await sess.run(new ort.Tensor(xa, [1, xa.length]));
        const diff = performance.now() - start;
        update_status(diff);

        // append results to textarea 
        setText((p) => { return `${p}${ret.str.data[0]}\n` })
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
      log(`total ${processing_time.toFixed(1)}sec for ${total.toFixed(1)}sec`);
      ready();
    }
  }

  const [ text, setText ] = useState("")

  return <>
    <div className="m-5 grid grid-flow-row grid-cols-3 grid-rows-3 gap-2 text-white border-gray-400 bg-black">
      <div className="">
        <div className="border col-md-4 rounded-block">
          <h4 className='bg-black'>Audio Input</h4>
          <button onClick={() => { isRecording ? stopRecord() : startRecord() }} disabled={ !sess || isLoading}
              className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded disabled:bg-gray-500">
            { isRecording ? "Stop Recording" : "Record" }
          </button>
        </div>
        <div className="border col-md-4 rounded-block">
          <h4 className='bg-black'>Transcribe</h4>
          <audio ref={(audio) => (audioRef.current = audio)} controls />
          <br />
          <button onClick={transcribe_file} disabled={ !sess || isLoading}
            className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded disabled:bg-gray-500">
              Get Text
          </button>
        </div>
        <div className="border col-md-4 rounded-block">
            <h4 className='bg-black'>Stats</h4>
            <div className="bg-gray-700">
                <div id="latency"></div>
            </div>
        </div>
      </div>
      <div className="gap-3">
        <textarea id="message" rows={12} 
          className="block p-2.5 w-full h-full text-sm text-gray-900 bg-gray-50 rounded-lg border border-gray-300 focus:ring-blue-500 focus:border-blue-500 dark:bg-black dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500" 
          placeholder="Text coming from OpenAI Whisper Model." value={text} onChange={() => {}}/>
        </div>
        <div className="border rounded-block gap-3">
          <div id="status" style={{font: "1em consolas", whiteSpace: "pre-wrap"}}>
            {logText}
          </div>
        </div>
    </div>
  </>
}

// wrapper around onnxruntime and model
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
          //"attention_mask": new ort.Tensor(new Int32Array(this.attention_mask), [1, 80, 3000]),
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
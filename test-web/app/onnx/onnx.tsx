"use client"

import * as ort from 'onnxruntime-web'
import { useEffect, useRef, useState } from 'react'

export default function Onnx() {

  const [inferenceSession, setInferenceSession] = useState<ort.InferenceSession|null>(null)

  const main = async () => {
    ort.env.wasm.wasmPaths = {
      'ort-wasm.wasm': '/models/onnx/dist/ort-wasm.wasm',
      'ort-wasm-simd.wasm': '/models/onnx/dist/ort-wasm-simd.wasm',
      'ort-wasm-threaded.wasm': '/models/onnx/dist/ort-wasm-threaded.wasm',
    }

    const opt = {
      executionProviders: ["wasm"],
      logSeverityLevel: 0,
      logVerbosityLevel: 0
    } as ort.InferenceSession.SessionOptions

    const session = await ort.InferenceSession.create('/models/onnx/v1/decoder_model.onnx', opt)
    console.log(session)
    setInferenceSession(session)
  }

  const run = async (audio_pcm: ort.TypedTensor<"float32">, beams = 1) => {
    if (!inferenceSession) return 

    const min_length = Int32Array.from({ length: 1 }, () => 1);
    const max_length = Int32Array.from({ length: 1 }, () => 448);
    const num_return_sequences = Int32Array.from({ length: 1 }, () => 1);
    const length_penalty = Float32Array.from({ length: 1 }, () => 1.);
    const repetition_penalty = Float32Array.from({ length: 1 }, () => 1.);
    const attention_mask = Int32Array.from({ length: 1 * 80 * 3000 }, () => 0);

    const feed = {
      "audio_pcm": audio_pcm,
      "max_length": new ort.Tensor(new Int32Array(max_length), [1]),
      "min_length": new ort.Tensor(new Int32Array(min_length), [1]),
      "num_beams": new ort.Tensor(Int32Array.from({ length: 1 }, () => beams), [1]),
      "num_return_sequences": new ort.Tensor(new Int32Array(num_return_sequences), [1]),
      "length_penalty": new ort.Tensor(new Float32Array(length_penalty), [1]),
      "repetition_penalty": new ort.Tensor(new Float32Array(repetition_penalty), [1]),
      "attention_mask": new ort.Tensor(new Int32Array(attention_mask), [1, 80, 3000]),
    }

    return await inferenceSession.run(feed)
  }

  useEffect(() => { 
    main()
    getMicrophone()
    return () => { closeMicrophone() }
  }, [])

  const [recorder, setRecorder] = useState<MediaRecorder|null>(null)
  const [audioBlob, setAudioBlob] = useState<Blob|null>(null)
  const [isRecording, setIsRecording] = useState(false)
  const [timerStart, setTimerStart] = useState<number>(0)

  const getMicrophone = () => {
    // needs to be SSL to work on other devices other than localhost
    navigator.mediaDevices.getUserMedia({
      audio: { 
        echoCancellation: false,
        noiseSuppression: false,
      }, video: false
    }).then((stream: MediaStream) => {
      console.log(`Audio Stream set -`)
      console.log(stream)
      var audioRecorder = new MediaRecorder(stream)
      setRecorder(audioRecorder)

      var chunks = [] as BlobPart[]
      if (audioRecorder) {
        audioRecorder.ondataavailable = (ev) => {
          chunks.push(ev.data)
        }

        audioRecorder.onstop = async ev => {
          let blob = new Blob(chunks, { type: 'audio/wav' })
          setAudioBlob(blob)
          chunks = [] as BlobPart[]
        }
      } else console.log("no audioRecorder!")
      
    }).catch(e => {
      console.log(e.name, e.message)
      //if (e.name )
    })
  }

  const closeMicrophone = () => {
    recorder && recorder.stop()
    recorder? recorder.ondataavailable = null : console.log("could not deregister recorder.ondataavailable")
    recorder? recorder.onstop = null : console.log("could not deregister recorder.onstop")
    setRecorder(null)
    console.log(`closed Microphone!`)
  }

  const startRecorder = () => {
    console.log(recorder)
    if (recorder && !isRecording) {
      recorder.start()
      setIsRecording(true)
      setTimerStart(performance.now())
      console.log("recording started")
    } else {
      console.warn("No Audio Recorder to start!")
    }
  }
  const stopRecorder = () => { 
    setIsRecording(false)
    recorder && recorder.stop()
    let time = performance.now() - timerStart
    console.log(time)
  }

  const audioRef = useRef<HTMLAudioElement | null>(null);

  useEffect(() => {
    if (audioRef.current && audioBlob) {
      const audioUrl = URL.createObjectURL(audioBlob);
      audioRef.current.src = audioUrl;
    }
  }, [audioBlob])

  const playAudio = () => {
    if (audioRef.current && audioBlob) {
      const audioUrl = URL.createObjectURL(audioBlob);
      audioRef.current.src = audioUrl;
      audioRef.current.play();
    }
  }

  const decode = async () => {
    if (inferenceSession && audioBlob) {
      const arrayBuffer = await readBlobAsArrayBuffer(audioBlob);
      const float32Array = await decodeAudioData(arrayBuffer)
      //const tens = new ort.Tensor("float32", float32Array, [float32Array.length])

      console.log(float32Array)

      //const result = inferenceSession.run({a: tens})
      const result = await run(new ort.Tensor(float32Array, [1, float32Array.length]))
      console.log(result)
    }
  }

  const readBlobAsArrayBuffer = (blob: Blob): Promise<ArrayBuffer> => {
    return new Promise((resolve, reject) => {
      const fileReader = new FileReader()
      fileReader.onload = (event) => {
        if (event.target && event.target.result) {
          resolve(event.target.result as ArrayBuffer)
        } else {
          reject(new Error('Failed to read blob as ArrayBuffer.'))
        }
      }
      fileReader.onerror = () => {
        reject(new Error('Error reading blob.'))
      }
      fileReader.readAsArrayBuffer(blob)
    })
  }

  const decodeAudioData = (arrayBuffer: ArrayBuffer): Promise<Float32Array> => {
    return new Promise((resolve, reject) => {
      const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
      audioContext.decodeAudioData(arrayBuffer, (buffer) => {
        resolve(buffer.getChannelData(0));
      }, (error) => {
        reject(error);
      });
    });
  }

  return <>
    <div className='my-5'>
      <button onClick={() => { isRecording ? stopRecorder() : startRecorder() }}
          className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded">
        { isRecording ? "Stop Recording" : "Start Recording" }
      </button>
    </div>
    { audioBlob && <div className='my-5'>
      <button onClick={playAudio} className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded">Play Audio</button>
      <audio ref={(audio) => (audioRef.current = audio)} controls />
    </div> }
    { inferenceSession && audioBlob && <div className='my-5'>
      <button onClick={decode} className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded">Get Text</button>
    </div> }
  </>
}
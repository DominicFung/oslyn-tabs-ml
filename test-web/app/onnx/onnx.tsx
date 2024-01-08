"use client"

import { InferenceSession } from 'onnxruntime-web'
import { useEffect } from 'react'

export default function Onnx() {

  const main = async () => {
    const session = await InferenceSession.create('../../../whisper/onnx/decoder_model.onnx')
    console.log(session)
  }

  useEffect(() => { main() }, [])

  return <>Onnx Page!</>
}
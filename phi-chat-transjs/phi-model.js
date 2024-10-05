import { addErrorMsg, addStatusMsg, showQueryControls } from './ui.js'

// ***************************************************************
// DOES NOT WORK YET!
// transformers.js doesn't support the model & external data yet
// ***************************************************************

// Using v3 of the Transformers library - still in alpha/beta
import { env, pipeline } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.0-alpha.19'

const MODEL = 'microsoft/Phi-3-mini-4k-instruct-onnx-web'
//const MODEL = 'Xenova/Phi-3-mini-4k-instruct'
const USE_LOCAL_MODEL = false

env.localModelPath = '../models'
env.allowRemoteModels = !USE_LOCAL_MODEL
env.allowLocalModels = USE_LOCAL_MODEL

let phiPipeline = null

export async function setUp() {
  try {
    phiPipeline = await pipeline('text-generation', MODEL, {
      device: 'webgpu',
      use_external_data_format: true,
      dtype: 'q4f16',
    })

    console.log(pipeline)

    addStatusMsg('🚀 Model loaded, session started!')
    showQueryControls()
  } catch (e) {
    addErrorMsg('' + e)
  }
}

export async function queryModel(query, id, continuation = false) {
  addStatusMsg(`🧠 Beginning query for "${query}"`)
}

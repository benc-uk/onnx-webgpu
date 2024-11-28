import { checkCache } from '../lib/utils.js'
import { addErrorMsg, addStatusMsg, showQueryControls, setResponseText } from './ui.js'

// ***************************************************************
// DOES NOT WORK YET!
// transformers.js doesn't support the model & external data yet
// https://github.com/xenova/transformers.js/issues/963
// ***************************************************************

// Using v3 of the Transformers library - still in alpha/beta
import { env, pipeline } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.2'

const MODEL = 'microsoft/Phi-3-mini-4k-instruct-onnx-web'
//const MODEL = 'onnx-community/MobileLLM-125M'
//const MODEL = 'Xenova/Phi-3-mini-4k-instruct'
//const MODEL = 'schmuell/Phi-3-mini-4k-instruct-onnx-web'
const USE_LOCAL_MODEL = false

env.localModelPath = '../models'
env.allowRemoteModels = !USE_LOCAL_MODEL
env.allowLocalModels = USE_LOCAL_MODEL

let phiPipeline = null

export async function setUp() {
  addStatusMsg('🧠 Loading model...')
  checkCache('../models/phi-3-mini-4k-instruct.onnx')
  try {
    phiPipeline = await pipeline('text-generation', MODEL, {
      device: 'webgpu',
      use_external_data_format: true,
      dtype: 'q4f16',
      // dtype: 'fp32',
    })

    // console.log(pipeline)

    addStatusMsg('🚀 Model loaded, session started!')
    showQueryControls()
  } catch (e) {
    addErrorMsg('' + e)
  }
}

export async function queryModel(query, id, continuation = false) {
  addStatusMsg(`🧠 Beginning query for "${query}"`)

  try {
    const response = await phiPipeline(query, { max_new_tokens: 100 })
    console.log(response)
    const responseText = response[0].generated_text
    addStatusMsg(`🎉 Query for "${query}" completed!`)

    setResponseText(id, responseText)
  } catch (e) {
    addErrorMsg('' + e)
  }
}

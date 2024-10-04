import * as utils from "../lib/utils.js"
import { addErrorMsg, addStatusMsg, setResponseText, showQueryControls } from "./ui.js"

import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/esm/ort.webgpu.min.js"
import { AutoTokenizer, env as transEnv } from "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.1"

const MODEL = "microsoft/Phi-3-mini-4k-instruct-onnx-web"
const USE_LOCAL_MODEL = false
const MAX_TOKENS = 1024

// Setup for transformers.js tokenizer
transEnv.localModelPath = "../models"
transEnv.allowRemoteModels = !USE_LOCAL_MODEL
transEnv.allowLocalModels = USE_LOCAL_MODEL

// Setup for ORT WASM path override
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/"

// ================================

let tokenizer
let ortSession
let modelConfig
let feed = {}
let stop = false

export async function setUp() {
  const capability = await utils.GetCapability()

  if (capability !== utils.WEBGPU_F16) {
    addErrorMsg("WebGPU not supported, or not supporting Float16")
    return
  }
  addStatusMsg("🔬 WebGPU and Float16 supported OK")

  try {
    tokenizer = await AutoTokenizer.from_pretrained(MODEL)
    addStatusMsg("📚 Tokenizer ready")

    const modelPath = USE_LOCAL_MODEL ? "../models/" + MODEL : "https://huggingface.co/" + MODEL + "/resolve/main"
    addStatusMsg(`📁 Model path: ${modelPath}`)
    const modelFile = "model_q4f16.onnx"
    const modelFqn = modelPath + "/onnx/" + modelFile
    console.log(modelFqn)

    // Load the model config
    addStatusMsg("🛠️ Loading model config...")
    const json_bytes = await utils.fetchAndCache(modelPath + "/config.json")
    let textDecoder = new TextDecoder()
    modelConfig = JSON.parse(textDecoder.decode(json_bytes))

    // Load the model weights
    addStatusMsg("📦 Loading ONNX model...")
    if (!(await utils.checkCache(modelFqn))) {
      addStatusMsg("⌛ Warning: file not in cache, downloading 800MB can take a while, please be patient...")
    }
    const modelBytes = await utils.fetchAndCache(modelFqn)

    // Load the external data as the phi-3 model has some
    addStatusMsg("💽 Loading model external data...")
    if (!(await utils.checkCache(modelPath + "/onnx/" + modelFile + "_data"))) {
      addStatusMsg("⌛ Warning: file not in cache, downloading 1.4GB can take a while, please be patient...")
    }
    const externalData = await utils.fetchAndCache(modelPath + "/onnx/" + modelFile + "_data")
    const modelSize = modelBytes.byteLength + externalData.byteLength

    addStatusMsg(`🛄 Total size ${Math.round(modelSize / 1024 / 1024)} MB`)
    addStatusMsg(`⏰ Starting ONNX Session...`)

    ortSession = await ort.InferenceSession.create(modelBytes, {
      executionProviders: ["webgpu"],
      preferredOutputLocation: {},
      externalData: [
        {
          data: externalData,
          path: modelFile + "_data",
        },
      ],
    })

    addStatusMsg("🚀 Model loaded, session started!")
    showQueryControls()
  } catch (e) {
    addErrorMsg("" + e)
  }
}

async function initFeedKV() {
  feed = {}

  const kvDims = [1, modelConfig.num_key_value_heads, 0, modelConfig.hidden_size / modelConfig.num_attention_heads]

  for (let i = 0; i < modelConfig.num_hidden_layers; ++i) {
    feed[`past_key_values.${i}.key`] = new ort.Tensor("float16", new Uint16Array(), kvDims)
    feed[`past_key_values.${i}.value`] = new ort.Tensor("float16", new Uint16Array(), kvDims)
  }
}

//
function updateFeedKV(outputs) {
  for (const name in outputs) {
    if (name.startsWith("present")) {
      let newName = name.replace("present", "past_key_values")

      const t = feed[newName]
      if (t.location === "gpu-buffer") {
        t.dispose()
      }

      feed[newName] = outputs[name]
    }
  }
}

export async function queryModel(query, id, continuation = false) {
  addStatusMsg(`🧠 Beginning query for "${query}"`)

  initFeedKV()
  stop = false

  const outputTokens = []
  const prompt = continuation ? query : `<|system|>\nYou are a friendly assistant.<|end|>\n<|user|>\n${query}<|end|>\n<|assistant|>\n`

  const { input_ids: rawTokens } = await tokenizer(prompt, {
    return_tensor: false,
    padding: true,
    truncation: true,
  })

  const inputIds = new ort.Tensor("int64", BigInt64Array.from(rawTokens.map(BigInt)), [1, rawTokens.length])
  feed["input_ids"] = inputIds

  // This is weird, but it's needed somehow
  outputTokens.push(...inputIds.data)

  let seqLen = outputTokens.length
  const inputLen = inputIds.size

  feed["position_ids"] = new ort.Tensor(
    "int64",
    BigInt64Array.from({ length: inputLen }, (_, i) => BigInt(seqLen - inputLen + i)),
    [1, inputLen]
  )

  let lastToken = 0n
  while (seqLen < MAX_TOKENS && lastToken != 32007 && lastToken != modelConfig.eos_token_id) {
    if (stop) {
      addStatusMsg("🛑 Stopping generation")
      break
    }

    seqLen = outputTokens.length
    feed["attention_mask"] = new ort.Tensor(
      "int64",
      BigInt64Array.from({ length: seqLen }, () => 1n),
      [1, seqLen]
    )

    const runOutput = await ortSession.run(feed)

    // Use argmax for some reason to get the right token
    lastToken = BigInt(utils.argmax(runOutput.logits))

    outputTokens.push(lastToken)

    const words = tokensToText(outputTokens, inputLen)
    setResponseText(id, words)

    updateFeedKV(runOutput)

    feed["input_ids"] = new ort.Tensor("int64", BigInt64Array.from([lastToken]), [1, 1])
    feed["position_ids"] = new ort.Tensor("int64", BigInt64Array.from([BigInt(seqLen)]), [1, 1])
  }
}

export function stopGeneration() {
  stop = true
}

function tokensToText(tokens, startIdx = 0) {
  const txt = tokenizer.decode(tokens.slice(startIdx), {
    skip_special_tokens: true,
  })

  return txt
}

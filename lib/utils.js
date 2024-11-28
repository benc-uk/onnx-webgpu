export const WEBGPU_F16 = 'webgpu-f16'
export const WEBGPU_NO_F16 = 'webgpu-no-f16'
export const NO_WEBGPU = 'no-webgpu'

// Check if the browser supports WebGPU and Float16
export async function GetCapability() {
  if (!('gpu' in navigator)) {
    return NO_WEBGPU
  }

  try {
    const adapter = await navigator.gpu.requestAdapter()
    console.dir(adapter)

    if (adapter.features.has('shader-f16')) {
      return WEBGPU_F16
    }

    return WEBGPU_NO_F16
  } catch (e) {
    return NO_WEBGPU
  }
}

// Check if a file is in cache
export async function checkCache(urlString) {
  const cache = await caches.open('onnx')

  // Parse the input URL string to solve relative URLs
  const url = new URL(urlString, location)
  const keys = await cache.keys()

  for (const request of keys) {
    if (request.url === url.href) {
      return true
    }
  }

  return false
}

// Fetches and caches files
// See https://developer.mozilla.org/en-US/docs/Web/API/Cache
export async function fetchAndCache(url) {
  console.log(`### Checking cache for: ${url}`)
  // Open and check cache
  const cache = await caches.open('onnx')
  let cachedResponse = await cache.match(url)

  // Fetch from network if not in cache
  if (cachedResponse === undefined) {
    const response = await fetch(url)
    const buffer = await response.arrayBuffer()

    await cache.put(url, new Response(buffer))

    console.log(`### ${url} Downloaded from network`)
    return buffer
  }

  console.log(`### ${url} Fetched from cache`)
  const data = await cachedResponse.arrayBuffer()
  return data
}

// Argmax function for logits tensor
// Ripped from https://github.com/microsoft/onnxruntime-inference-examples/blob/main/js/chat/llm.js
export function logitArgmax(t) {
  const arr = t.data
  const start = t.dims[2] * (t.dims[1] - 1)
  let max = arr[start]
  let maxidx = 0

  for (let i = 0; i < t.dims[2]; i++) {
    const val = arr[i + start]

    if (!isFinite(val)) {
      throw new Error('found infinitive in logits')
    }

    if (val > max) {
      max = arr[i + start]
      maxidx = i
    }
  }

  return maxidx
}

// Generate a random ID from a string
export function randIdFromString(str) {
  const strBytes = new TextEncoder().encode(str)
  const hashBuffer = crypto.subtle.digest('SHA-256', strBytes)
  const hashArray = Array.from(new Uint8Array(hashBuffer))
  const hashHex = hashArray.map((b) => b.toString(16).padStart(2, '0')).join('')

  return hashHex
}

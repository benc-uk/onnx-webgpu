export default {
  server: {
    headers: {
      // Trying to fix this warning: "env.wasm.numThreads is set to 4, but this will not work unless you enable crossOriginIsolated mode."
      // This seemed to be a good idea, but enabling it results in "no available backend found" error
      //'Cross-Origin-Embedder-Policy': 'require-corp',
      //'Cross-Origin-Opener-Policy': 'same-origin',
    },
  },
}

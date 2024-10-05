import { setUp } from "./phi-model.js"
import { initUI } from "./ui.js"

window.addEventListener("load", () => {
  initUI()
  setUp()
})

import { queryModel, stopGeneration } from "./phi-model.js"
import * as utils from "../lib/utils.js"

const statusArea = document.getElementById("status")
const sendButton = document.getElementById("send-button")
const queryTA = document.getElementById("user-input")
const stopButton = document.getElementById("stop-button")
const responses = document.getElementById("responses")

export function initUI() {
  sendButton.addEventListener("click", sendQuery)
  stopButton.addEventListener("click", stopGeneration)
}

export function addStatusMsg(...status) {
  statusArea.innerHTML += status + "<br>"
  statusArea.scrollTop = statusArea.scrollHeight
}

export function addErrorMsg(error) {
  statusArea.innerHTML += error + "<br>"
  statusArea.classList.add("error-text")
  statusArea.scrollTop = statusArea.scrollHeight
}

export function setResponseText(id, words) {
  document.getElementById(`resp-text-${id}`).innerText = words
}

export function showQueryControls() {
  queryTA.classList.remove("hidden")
  sendButton.classList.remove("hidden")
}

export function hideQueryControls() {
  queryTA.classList.add("hidden")
  sendButton.classList.add("hidden")
}

export function addResponseCard(id, query) {
  const cardDiv = document.createElement("div")

  cardDiv.innerHTML = `<div class="card" id="resp-card" readonly="true">
  <h2>💬 ${query}</h2>
  <p id="resp-text-${id}">Please wait, I am thinking how to answer that...</p></div>`

  responses.prepend(cardDiv)
}

async function sendQuery() {
  if (!queryTA.value) return

  hideQueryControls()
  stopButton.classList.remove("hidden")

  const queryID = utils.randIdFromString(queryTA.value)
  addResponseCard(queryID, queryTA.value)

  await queryModel(queryTA.value, queryID)

  showQueryControls()
  stopButton.classList.add("hidden")
}

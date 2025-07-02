import SimplePeer from 'simple-peer';
import io from 'socket.io-client';
import QRCode from 'qrcode';

let peers = [];
let p2pActive = false;
let externalCallback = null;
let socket;

export function initP2PFallback(onMessage) {
  externalCallback = onMessage;
  socket = io('http://localhost:3001');
  const peer = new SimplePeer({ initiator: true, trickle: false });

  peer.on('signal', data => {
    showQRCode(JSON.stringify(data));
    socket.emit('signal', { to: 'all', data });
  });

  socket.on('signal', ({ from, data }) => {
    peer.signal(data);
  });

  peer.on('connect', () => {
    p2pActive = true;
    updateDiagnostics('Mesh Connected');
  });

  peer.on('data', data => {
    const msg = JSON.parse(data);
    if (msg.type === 'broadcast' && externalCallback) {
      externalCallback(msg.payload);
      updateDiagnostics('Msg @ ' + new Date().toLocaleTimeString());
    }
  });

  peers.push(peer);
}

export function sendP2PBroadcast(payload) {
  peers.forEach(peer => {
    if (peer.connected) {
      peer.send(JSON.stringify({ type: 'broadcast', payload }));
    }
  });
}

export function isP2PActive() {
  return p2pActive;
}

export function manualSignalInput(input) {
  try {
    peers[0].signal(JSON.parse(input));
    updateDiagnostics('Manual signal accepted');
  } catch (e) {
    alert('Invalid signal input');
  }
}

function showQRCode(text) {
  const canvas = document.getElementById('qr-code');
  if (canvas) QRCode.toCanvas(canvas, text, err => {
    if (err) console.error(err);
  });
  const rawOut = document.getElementById('signal-raw');
  if (rawOut) rawOut.textContent = text;
}

function updateDiagnostics(status) {
  const statusEl = document.getElementById('mesh-mode');
  const lastMsgEl = document.getElementById('last-msg');
  if (statusEl) statusEl.textContent = isP2PActive() ? 'Connected' : 'Polling';
  if (lastMsgEl) lastMsgEl.textContent = status;
}
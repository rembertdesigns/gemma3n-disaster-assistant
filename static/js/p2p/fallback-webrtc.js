// static/js/p2p/fallback-webrtc.js

import SimplePeer from 'simple-peer';
import io from 'socket.io-client';
import QRCode from 'qrcode';

let peers = [];
let p2pActive = false;
let externalCallback = null;
let socket;
let manualPeer = null;
let qrRendered = false;

// Initialize WebRTC with Socket.io signaling
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

// Manual pairing offer generator
export async function manualSignalOffer() {
  manualPeer = new SimplePeer({ initiator: true, trickle: false });

  manualPeer.on('signal', async data => {
    const signalString = JSON.stringify(data);
    const qrCanvas = document.getElementById('qr-code');
    const signalOut = document.getElementById('signal-raw');

    if (signalOut) signalOut.textContent = signalString;
    if (qrCanvas && !qrRendered) {
      try {
        await QRCode.toCanvas(qrCanvas, signalString);
        qrRendered = true;
      } catch (err) {
        console.error('QR generation failed', err);
      }
    }
  });

  manualPeer.on('connect', () => {
    p2pActive = true;
    updateDiagnostics('Manual P2P Connected');
    console.log('✅ Manual P2P connection established.');
  });

  manualPeer.on('data', data => {
    const msg = JSON.parse(data);
    if (msg.type === 'broadcast' && externalCallback) {
      externalCallback(msg.payload);
      updateDiagnostics('Msg @ ' + new Date().toLocaleTimeString());
    }
  });

  peers.push(manualPeer);
}

// Accept a manual signal from another peer
export function manualSignalInput(input) {
  try {
    const data = JSON.parse(input);
    if (manualPeer) {
      manualPeer.signal(data);
      updateDiagnostics('Manual signal accepted');
      console.log('📡 Manual signal applied');
    } else {
      alert('❌ No manual peer instance found');
    }
  } catch (e) {
    alert('❌ Invalid signal input');
  }
}

// Broadcast a message to all connected peers
export function sendP2PBroadcast(payload) {
  peers.forEach(peer => {
    if (peer.connected) {
      peer.send(JSON.stringify({ type: 'broadcast', payload }));
    }
  });
}

// Check if mesh mode is active
export function isP2PActive() {
  return p2pActive;
}

// Internal: Display QR code and raw signal
function showQRCode(text) {
  const canvas = document.getElementById('qr-code');
  if (canvas) QRCode.toCanvas(canvas, text, err => {
    if (err) console.error(err);
  });

  const rawOut = document.getElementById('signal-raw');
  if (rawOut) rawOut.textContent = text;
}

// Internal: Update connection + diagnostics UI
function updateDiagnostics(status) {
  const statusEl = document.getElementById('mesh-mode');
  const lastMsgEl = document.getElementById('last-msg');
  if (statusEl) statusEl.textContent = isP2PActive() ? 'Connected' : 'Polling';
  if (lastMsgEl) lastMsgEl.textContent = status;
}
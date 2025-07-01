let peers = [];
let localStream = null;

// Initialize WebRTC fallback
export function initP2PFallback() {
  console.log("🌐 Initializing P2P fallback...");

  // This peer initiates connection (use signaling in future)
  const peer = new SimplePeer({ initiator: true, trickle: false });

  peer.on('signal', data => {
    // ⚠️ Demo-only: You'd normally use a signaling server or QR/Bluetooth
    console.log('🔑 Share this signal with peer:', JSON.stringify(data));
  });

  peer.on('connect', () => {
    console.log('✅ P2P connection established');
    peer.send(JSON.stringify({ type: "ping", time: Date.now() }));
  });

  peer.on('data', data => {
    const msg = JSON.parse(data);
    console.log("📡 Received via P2P:", msg);

    // Dispatch broadcast to app if it's an alert
    if (msg.type === "broadcast") {
      triggerLocalBroadcast(msg.payload);
    }
  });

  peers.push(peer);
}

// Simulated broadcast payload sender
export function sendP2PBroadcast(payload) {
  peers.forEach(peer => {
    if (peer.connected) {
      peer.send(JSON.stringify({ type: "broadcast", payload }));
      console.log("📤 Sent broadcast via P2P");
    }
  });
}

// Trigger UI update on local broadcast
function triggerLocalBroadcast(data) {
  const alert = document.getElementById("broadcast-alert");
  if (alert) alert.style.display = "block";

  const feed = document.getElementById("broadcast-feed");
  const newCard = `
    <div class="broadcast-card">
      <p><strong>🚨 ${data.severity}</strong></p>
      <p>${data.message}</p>
      <small><code>Local P2P · ${new Date().toISOString()}</code></small>
      <hr/>
    </div>`;
  feed.innerHTML = newCard + feed.innerHTML;
}
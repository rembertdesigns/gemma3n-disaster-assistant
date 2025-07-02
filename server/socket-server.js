import express from 'express';
import http from 'http';
import { Server as SocketIO } from 'socket.io';

const app = express();
const server = http.createServer(app);
const io = new SocketIO(server, {
  cors: { origin: '*' }
});

io.on('connection', socket => {
  socket.on('join-room', ({ room, id }) => {
    socket.join(room);
    socket.to(room).emit('peer-connected', { id });

    socket.on('signal', ({ to, data }) => {
      io.to(to).emit('signal', { from: socket.id, data });
    });
  });
});

server.listen(3001, () => {
  console.log('🚦 Signaling server running on http://localhost:3001');
});
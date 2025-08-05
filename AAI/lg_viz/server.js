// server.js
const express = require('express');
const app = express();
const http = require('http').createServer(app);
const io = require('socket.io')(http);
const path = require('path');

// Middleware to parse JSON bodies
app.use(express.json());

// Serve static files from public/
app.use(express.static(path.join(__dirname, 'public')));

// Endpoint to clear logs/graph. Called by the streaming function as needed.
app.post('/api/clear', (req, res) => {
  console.log(`[Server] Received clear`);
  // Emit the clear event to all connected clients
  io.emit('clear', {});
  res.json({ status: 'ok' });
});

// Endpoint for receiving updates from the Python streaming function.
app.post('/api/update', (req, res) => {
  // Expect JSON payload with at least { mode, data }
  const { mode, data } = req.body;
  console.log(`[Server] Received update - Mode: ${mode}`);
  // Emit the update event to all connected clients
  io.emit('update', { mode, data });
  res.json({ status: 'ok' });
});

// Log socket connection/disconnection events
io.on('connection', (socket) => {
  console.log('A client connected: ' + socket.id);
  socket.on('disconnect', () => {
    console.log('Client disconnected: ' + socket.id);
  });
});

// Start the server
const PORT = process.env.PORT || 3002;
http.listen(PORT, () => {
  console.log(`Server is listening on port ${PORT}`);
});

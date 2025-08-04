# Dockerfile for the visualizer application
# Base image
FROM node:20-alpine

# Set working directory
WORKDIR /app

COPY package*.json ./
RUN npm install --verbose

# Expose port
EXPOSE 3002
COPY . .

# Start the application
ENTRYPOINT ["node", "server.js"]

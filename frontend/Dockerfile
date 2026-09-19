# Use Node.js as the base image for building the frontend
FROM node:18-alpine AS builder

WORKDIR /app

# Install dependencies
COPY package.json package-lock.json* ./
RUN npm install

# Copy application files
COPY . .

# Expose Vite's default dev server port
EXPOSE 5173

# Start Vite dev server for easy development (production could use nginx)
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]

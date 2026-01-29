FROM node:20-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --omit=dev || npm install --omit=dev
COPY server.js ./
ENV NODE_ENV=production PORT=5001 TZ=Asia/Taipei
EXPOSE 5001
CMD ["node","server.js"]

const path=require('path');
const express=require('express');
const app=express()
const server=require('http').Server(app)
const io=require('socket.io')(server)
const cv=require('opencv4nodejs')
// const app=express();

const camImg=new cv.VideoCapture(0);

// let processedFrame;
const FPS=30;
const PORT=3000;
// const pathImg=path.join(__dirname, 'index.html');

app.get('/', (req, res)=>{
  res.sendFile(path.join(__dirname, 'index.html'));
})

setInterval(()=>{
  const frame=camImg.read();
  const img=cv.imencode('.jpg', frame).toString('base64')
  console.log(img)
  io.emit('image', img)
}, 1000/FPS)

server.listen(PORT)

// const process = spawn('python', ['./mask_detector.py', 0])

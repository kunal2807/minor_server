/*

STEPS

start=>call MASK_DETECTOR.py=>
=>(start_camera=>get_frames=>process_frame=>put_processed_frame_in_stdout_pipeline)=>   //in python//
=>recived_frame_buffer_on_'data'_variable=>encode_to_jpg=>emit_img_via_socket

*/


const path=require('path'); 
const express=require('express');
const app=express()
const server=require('http').Server(app)
const io=require('socket.io')(server)
// let encode = require('image-encode')
// const cv=require('opencv4nodejs');
// const { type } = require('os');
// const { exit } = require('process');
const spawn = require('child_process').spawn; // To Spawn Python Code
// var fs=require('fs')
// const btoa = require('window').btoa
// const app=express();

// const camImg=new cv.VideoCapture(0);

// let processedFrame;
// const FPS=30;
const PORT=3000;
// const BUFFER_SIZE = 1000;

// const pathImg=path.join(__dirname, 'index.html');

app.get('/', (req, res)=>{
  res.sendFile(path.join(__dirname, 'index.html')); //To send to index.html:: NOTE: nothing to do with python
})

console.log("spawning")
const process = spawn('python', ['./MASK_DETECTOR.py']) //CALL PYTHON PROCESS
console.log("spawned")

//using stdout pipeline method
//on dumping some data to stdout (from python)
//Following snippet run
//data=buffer of frames
process.stdout.on('data', data=>{
  var frame = data.toString('base64') //convert buffer to base64 encoded string
  // io.emit('image', frame);
  // console.log("frame: ", frame) //log a frame (nothing but a jpg image)
  // console.log("data: ", data) //log buffer of image
  
  // const frame =data.toString()
  // console.log(frame)
  // var frame = Buffer.from(encode(data, [2,2], 'jpg'));
  console.log('OUTPUTFRAME: ', frame)
  console.log('OUTPUT DATA: ', data)
  io.emit('image', frame) // emit the image to index.html file (NOTHING TO DO WITH PYTHON)
})

// io.on('connection', function (socket) {
//   socket.on('data', function (data) {                     // listen on client emit 'data'
//     var frame = Buffer.from(data, 'base64').toString()
//     io.emit('image', frame);                                 // emmit to socket
//   })
// })

server.listen(PORT)

// const process = spawn('python', ['./mask_detector.py', 0])


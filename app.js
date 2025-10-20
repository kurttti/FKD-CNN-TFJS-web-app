let model=null;
const inputCanvas=document.getElementById('inputCanvas');
const overlayCanvas=document.getElementById('overlayCanvas');
const ctxIn=inputCanvas.getContext('2d'); const ctxOut=overlayCanvas.getContext('2d');
const statusEl=document.getElementById('status');

// Test if model files are accessible
async function testModelAccess() {
  try {
    console.log('Testing model file access...');
    const response = await fetch('model/model.json');
    if (response.ok) {
      console.log('Model file is accessible');
      return true;
    } else {
      console.error('Model file not accessible:', response.status);
      return false;
    }
  } catch (error) {
    console.error('Error accessing model file:', error);
    return false;
  }
}
document.getElementById('loadModelBtn').addEventListener('click',async()=>{
  try {
    statusEl.textContent='Testing model access...';
    console.log('Starting model load...');
    
    // First test if model files are accessible
    const isAccessible = await testModelAccess();
    if (!isAccessible) {
      statusEl.textContent='Model files not accessible. Check GitHub Pages deployment.';
      return;
    }
    
    statusEl.textContent='Loading model... (This may take 1-2 minutes for 20MB model)';
    
    // Add progress indicator
    let progressCount = 0;
    const progressInterval = setInterval(() => {
      progressCount++;
      statusEl.textContent=`Loading model... (${progressCount}s) - Large model, please wait...`;
    }, 1000);
    
    // Add longer timeout for large model files (20MB total)
    const loadPromise = tf.loadLayersModel('model/model.json');
    const timeoutPromise = new Promise((_, reject) => 
      setTimeout(() => reject(new Error('Model loading timeout after 2 minutes')), 120000)
    );
    
    model = await Promise.race([loadPromise, timeoutPromise]);
    clearInterval(progressInterval);
    console.log('Model loaded successfully');
    statusEl.textContent='Model loaded successfully!';
  } catch (error) {
    clearInterval(progressInterval);
    console.error('Error loading model:', error);
    statusEl.textContent='Error loading model: ' + error.message;
    
    // Try to provide more specific error information
    if (error.message.includes('404')) {
      statusEl.textContent='Model file not found. Check if model.json exists.';
    } else if (error.message.includes('CORS')) {
      statusEl.textContent='CORS error. Try running from a local server.';
    } else if (error.message.includes('timeout')) {
      statusEl.textContent='Model loading timed out. File might be too large.';
    }
  }
});
document.getElementById('imageInput').addEventListener('change',async(e)=>{
  const file=e.target.files[0]; if(!file) return; await drawAndPredict(file);
});
async function drawAndPredict(file){
  if(!model){statusEl.textContent='Load the model first.';return;}
  try {
    statusEl.textContent='Processing image...';
    const img=await fileToImage(file); 
    ctxIn.clearRect(0,0,96,96); 
    ctxIn.drawImage(img,0,0,96,96);
    const imgData=ctxIn.getImageData(0,0,96,96); 
    const data=imgData.data; 
    const gray=new Float32Array(96*96);
    for(let i=0,j=0;i<data.length;i+=4,j++){ 
      gray[j]=(0.299*data[i]+0.587*data[i+1]+0.114*data[i+2])/255.0; 
    }
    statusEl.textContent='Running prediction...';
    const t=tf.tensor(gray,[1,96,96,1]); 
    const y=model.predict(t); 
    const coords=(await y.array())[0]; 
    y.dispose(); 
    t.dispose();
    ctxOut.clearRect(0,0,96,96); 
    ctxOut.drawImage(inputCanvas,0,0); 
    ctxOut.strokeStyle='#38bdf8';
    for(let k=0;k<30;k+=2){ 
      drawCross(ctxOut, coords[k], coords[k+1]); 
    } 
    statusEl.textContent='Prediction complete!';
  } catch (error) {
    console.error('Error during prediction:', error);
    statusEl.textContent='Error during prediction: ' + error.message;
  }
}
function drawCross(ctx,x,y){ const r=2; ctx.beginPath(); ctx.moveTo(x-r,y); ctx.lineTo(x+r,y); ctx.moveTo(x,y-r); ctx.lineTo(x,y+r); ctx.stroke(); }
function fileToImage(file){ return new Promise((res)=>{ const img=new Image(); img.onload=()=>res(img); img.src=URL.createObjectURL(file); }); }

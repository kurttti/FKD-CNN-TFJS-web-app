let model=null;
const inputCanvas=document.getElementById('inputCanvas');
const overlayCanvas=document.getElementById('overlayCanvas');
const ctxIn=inputCanvas.getContext('2d'); const ctxOut=overlayCanvas.getContext('2d');
const statusEl=document.getElementById('status');
document.getElementById('loadModelBtn').addEventListener('click',async()=>{
  try {
    statusEl.textContent='Loading model...';
    console.log('Starting model load...');
    model = await tf.loadLayersModel('model/model.json');
    console.log('Model loaded successfully');
    statusEl.textContent='Model loaded successfully!';
  } catch (error) {
    console.error('Error loading model:', error);
    statusEl.textContent='Error loading model: ' + error.message;
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
